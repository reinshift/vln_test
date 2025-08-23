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

# GroundingDINO imports (assuming they are installed in the venv)
# Note: This is a placeholder for the actual model loading and inference logic.
# The specific imports will depend on the GroundingDINO repository's structure.
# from GroundingDINO.groundingdino.util.inference import load_model, predict

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
        self.image_sub = rospy.Subscriber('/magv/camera/image_compressed', CompressedImage, self.image_callback, queue_size=1)
        self.prompt_sub = rospy.Subscriber('/grounding_dino/prompt', String, self.prompt_callback, queue_size=1)
        self.detections_pub = rospy.Publisher('/grounding_dino/detections', Detection2DArray, queue_size=10)

        # --- State Variables ---
        self.bridge = CvBridge()
        self.current_prompt = "" # The object class to detect
        self.model = None

        # --- Load Model ---
        # self.load_dino_model()
        rospy.logwarn("DINO MODEL LOADING IS A PLACEHOLDER. Inference will be simulated.")

        rospy.loginfo("GroundingDINO Node ready.")

    def load_dino_model(self):
        try:
            # This is where the actual model loading would happen.
            # self.model = load_model(self.model_path, self.device)
            rospy.loginfo(f"GroundingDINO model loaded successfully on {self.device}")
        except Exception as e:
            rospy.logerr(f"Failed to load GroundingDINO model: {e}")
            self.model = None

    def prompt_callback(self, msg):
        self.current_prompt = msg.data.strip().lower()
        rospy.loginfo(f"Received new detection prompt: '{self.current_prompt}'")

    def image_callback(self, msg):
        if not self.current_prompt:
            return # Don't process images if we don't have a prompt

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Failed to decompress image: {e}")
            return

        # --- Perform Inference (Simulated) ---
        # In a real implementation, you would call the model's predict function here.
        # boxes, logits, phrases = predict(
        #     model=self.model, 
        #     image=cv_image, 
        #     caption=self.current_prompt, 
        #     box_threshold=self.box_threshold, 
        #     text_threshold=self.text_threshold,
        #     device=self.device
        # )
        
        # --- Create Detection Messages (Simulated) ---
        # This part simulates a detection if the prompt is 'tree' for demonstration.
        detections_msg = Detection2DArray()
        detections_msg.header = msg.header
        if 'tree' in self.current_prompt:
            h, w, _ = cv_image.shape
            for _ in range(random.randint(1, 2)): # Simulate finding 1 or 2 trees
                detection = Detection2D()
                detection.header = msg.header
                detection.id = "0"
                detection.label = 'tree'
                detection.score = random.uniform(0.6, 0.95)
                
                box = BoundingBox2D()
                box.size_x = random.uniform(50, 150)
                box.size_y = random.uniform(100, 250)
                box.center.x = random.uniform(box.size_x / 2, w - box.size_x / 2)
                box.center.y = random.uniform(box.size_y / 2, h - box.size_y / 2)
                detection.bbox = box
                detections_msg.detections.append(detection)
        
        if detections_msg.detections:
            self.detections_pub.publish(detections_msg)

if __name__ == '__main__':
    import random # for simulation
    try:
        node = GroundingDinoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

