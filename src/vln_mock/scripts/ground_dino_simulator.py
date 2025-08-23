#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import random
import math
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from magv_vln_msgs.msg import DetectedObject, DetectedObjectArray

class GroundDinoSimulator:
    """
    This node simulates the behavior of a GroundDino object detector.
    It periodically publishes a list of detected objects with their labels and positions
    to the /detected_objects topic.
    """
    def __init__(self):
        rospy.loginfo("GroundDino Simulator Initializing...")

        # Publisher for detected objects
        self.objects_pub = rospy.Publisher('/detected_objects', DetectedObjectArray, queue_size=10)

        # Subscriber for robot odometry
        self.odom_sub = rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback, queue_size=1)

        # State
        self.robot_pose = None

        # List of possible objects to detect, as specified
        self.possible_objects = [
            'barrel', 'bench', 'billboard', 'fire hydrant', 
            'tractor trailer', 'traffic cone', 'trash bin', 'tree'
        ]

        # Simulation parameters
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)  # Hz
        self.detection_radius = rospy.get_param('~detection_radius', 15.0) # meters
        self.num_objects_to_spawn = rospy.get_param('~num_objects', 5)

        # Spawn a fixed set of objects for consistent simulation
        self.spawned_objects = self.spawn_initial_objects()

        # Timer to publish detections
        self.publish_timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_detections)

        rospy.loginfo(f"GroundDino Simulator started. Publishing {len(self.spawned_objects)} objects periodically.")

    def spawn_initial_objects(self):
        """Create a fixed set of objects in the environment at startup."""
        objects = []
        for i in range(self.num_objects_to_spawn):
            obj = DetectedObject()
            obj.header.frame_id = "map"
            obj.label = random.choice(self.possible_objects)
            obj.position.x = random.uniform(-self.detection_radius, self.detection_radius)
            obj.position.y = random.uniform(-self.detection_radius, self.detection_radius)
            obj.position.z = 0.2  # Assume objects are on the ground
            obj.confidence = random.uniform(0.7, 0.99)
            objects.append(obj)
            rospy.loginfo(f"Spawning '{obj.label}' at (x={obj.position.x:.2f}, y={obj.position.y:.2f})")
        return objects

    def odometry_callback(self, msg):
        """Store the latest robot pose."""
        self.robot_pose = msg.pose.pose

    def publish_detections(self, event):
        """Publish the list of objects visible to the robot."""
        if self.robot_pose is None:
            return

        detection_array = DetectedObjectArray()
        detection_array.header.stamp = rospy.Time.now()
        detection_array.header.frame_id = "map"

        visible_objects = []
        for obj in self.spawned_objects:
            dist_sq = (obj.position.x - self.robot_pose.position.x)**2 + \
                      (obj.position.y - self.robot_pose.position.y)**2
            if dist_sq < self.detection_radius**2:
                visible_objects.append(obj)

        detection_array.objects = visible_objects
        self.objects_pub.publish(detection_array)

if __name__ == '__main__':
    try:
        rospy.init_node('ground_dino_simulator', anonymous=True)
        simulator = GroundDinoSimulator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

