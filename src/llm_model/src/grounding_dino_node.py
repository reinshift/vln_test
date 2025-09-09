#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import rospy
import os
import torch
import numpy as np
from PIL import Image
import io
import cv2

# ROS Messages
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from magv_vln_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray

# GroundingDINO imports（原生库改为可选）
try:
    from groundingdino.util.inference import predict  # 仅在可用时用于旧分支
    HAS_GDINO_NATIVE = True
except Exception:
    HAS_GDINO_NATIVE = False

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDinoNode:
    def __init__(self):
        rospy.init_node('grounding_dino_node', anonymous=True)
        rospy.loginfo("Initializing GroundingDINO Node...")

        # --- Parameters ---
        # 允许通过 ~model_path 覆盖模型路径，默认指向包内 models/GroundingDino/grounding-dino-base
        default_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'GroundingDino', 'grounding-dino-base')
        self.model_path = rospy.get_param('~model_path', default_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.box_threshold = rospy.get_param('~box_threshold', 0.35)
        self.text_threshold = rospy.get_param('~text_threshold', 0.25)

        # --- ROS Subscribers & Publishers ---
        # Subscribe to the correct compressed image topic from rosbag
        self.image_sub = rospy.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage, self.image_callback, queue_size=1)
        self.prompt_sub = rospy.Subscriber('/grounding_dino/prompt', String, self.prompt_callback, queue_size=1)
        self.detections_pub = rospy.Publisher('/grounding_dino/detections', Detection2DArray, queue_size=10)

        # --- State Variables ---
        self.current_prompt = "" # The object class to detect
        self.model = None

        # --- Load Model ---
        self.load_dino_model()

        rospy.loginfo("GroundingDINO Node ready.")

    def load_dino_model(self):
        try:
            # 使用 HuggingFace Transformers 风格加载（本地目录或模型ID均可）
            resolved_path = os.path.abspath(self.model_path) if os.path.exists(self.model_path) else self.model_path
            rospy.loginfo(f"Loading HF Grounding DINO from: {resolved_path}")

            # 加载处理器与模型
            self.processor = AutoProcessor.from_pretrained(resolved_path)
            hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(resolved_path)
            self.model = hf_model.to(self.device)
            self.is_hf_model = True

            rospy.loginfo(f"HF Grounding DINO loaded successfully on {self.device}")
        except Exception as e:
            rospy.logerr(f"Failed to load GroundingDINO model: {e}")
            self.model = None

    def prompt_callback(self, msg):
        self.current_prompt = msg.data.strip()
        if self.current_prompt:
            rospy.loginfo(f"Received new detection prompt: '{self.current_prompt}'")
        else:
            rospy.loginfo("Prompt cleared.")

    def image_callback(self, msg):
        if not self.current_prompt or not self.model:
            return # Don't process images if we don't have a prompt or model

        try:
            image_rgb = Image.open(io.BytesIO(msg.data))
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")
            return

        # --- Perform Inference ---
        try:
            if hasattr(self, 'is_hf_model') and getattr(self, 'is_hf_model'):
                # HF 推理分支
                prompt = self.current_prompt.strip().lower()
                if prompt and not prompt.endswith('.'):  # HF要求以句号结尾
                    prompt += '.'

                inputs = self.processor(images=image_rgb, text=prompt, return_tensors="pt")
                inputs = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # PIL -> size (W,H) 获取
                w_img, h_img = image_rgb.size
                target_sizes = torch.tensor([(h_img, w_img)], device=self.device)
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.get('input_ids'),
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes
                )[0]

                boxes_xyxy = results.get('boxes')  # shape [N,4] xyxy in absolute pixels
                scores = results.get('scores')     # shape [N]
                labels = results.get('labels', []) # list[str]

                if boxes_xyxy is None or scores is None:
                    return

                # 转为与原始代码兼容的归一化 cx,cy,w,h 与 phrases/logits
                boxes_list = []
                for b in boxes_xyxy:
                    x1, y1, x2, y2 = b.tolist()
                    cx = (x1 + x2) / 2.0 / w_img
                    cy = (y1 + y2) / 2.0 / h_img
                    bw = (x2 - x1) / w_img
                    bh = (y2 - y1) / h_img
                    boxes_list.append(torch.tensor([cx, cy, bw, bh], dtype=torch.float32))

                boxes = boxes_list
                logits = scores
                phrases = labels if isinstance(labels, list) else []
            else:
                # 旧 GroundingDINO 推理分支
                if HAS_GDINO_NATIVE:
                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=image_rgb,
                        caption=self.current_prompt,
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        device=self.device
                    )
                else:
                    rospy.logerr("GroundingDINO native inference not available and HF model not loaded; skipping.")
                    return
        except Exception as e:
            rospy.logerr(f"Inference failed: {e}")
            return

        # --- Create Detection Messages ---
        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        if phrases:
            rospy.loginfo(f"Detected {len(phrases)} instances of '{', '.join(set(phrases))}'")

        for i in range(len(boxes)):
            detection = Detection2D()
            detection.header = msg.header
            detection.id = str(i)
            detection.label = phrases[i]
            detection.score = logits[i].item()

            # Convert box format from [cx, cy, w, h] (normalized) to BoundingBox2D
            w, h = image_rgb.size
            box_tensor = boxes[i]
            center_x = box_tensor[0] * w
            center_y = box_tensor[1] * h
            size_x = box_tensor[2] * w
            size_y = box_tensor[3] * h

            box_msg = BoundingBox2D()
            box_msg.center.x = center_x
            box_msg.center.y = center_y
            box_msg.size_x = size_x
            box_msg.size_y = size_y
            detection.bbox = box_msg

            detections_msg.detections.append(detection)

        if detections_msg.detections:
            self.detections_pub.publish(detections_msg)

if __name__ == '__main__':
    try:
        node = GroundingDinoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass