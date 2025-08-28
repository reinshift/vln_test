#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from magv_vln_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray

# GroundingDINO imports
from groundingdino.util.inference import load_model, load_image, predict, annotate

class GroundingDinoNode:
    def __init__(self):
        rospy.init_node('grounding_dino_node', anonymous=True)
        rospy.loginfo("Initializing GroundingDINO Node...")

        # --- Parameters ---
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'GroundingDino', 'grounding-dino-base')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.box_threshold = rospy.get_param('~box_threshold', 0.35)
        self.text_threshold = rospy.get_param('~text_threshold', 0.25)

        # --- ROS Subscribers & Publishers ---
        # Subscribe to the correct compressed image topic from rosbag
        self.image_sub = rospy.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage, self.image_callback, queue_size=1)
        self.prompt_sub = rospy.Subscriber('/grounding_dino/prompt', String, self.prompt_callback, queue_size=1)
        self.detections_pub = rospy.Publisher('/grounding_dino/detections', Detection2DArray, queue_size=10)

        # --- State Variables ---
        self.bridge = CvBridge()
        self.current_prompt = "" # The object class to detect
        self.model = None

        # --- Load Model ---
        self.load_dino_model()

        rospy.loginfo("GroundingDINO Node ready.")

    def load_dino_model(self):
        try:
            # This is where the actual model loading would happen.
            self.model = load_model(self.model_path, self.device)
            rospy.loginfo(f"GroundingDINO model loaded successfully on {self.device}")
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
            # Decompress image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Convert to RGB for GroundingDINO
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")
            return

        # --- Perform Inference ---
        try:
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_rgb,
                caption=self.current_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
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
            h, w, _ = cv_image.shape
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

