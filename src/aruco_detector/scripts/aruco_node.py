#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf.transformations as tfs
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

# Custom Messages
from aruco_detector.msg import ArucoInfo, ArucoMarker

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector_node', anonymous=True)
        rospy.loginfo("ArUco Detector Node Initializing...")

        # --- Parameters ---
        self.marker_size = rospy.get_param('~marker_size', 0.1)  # ArUco码的物理边长 (米)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # --- Camera Intrinsics ---
        image_width = rospy.get_param('~image_width', 1080)
        image_height = rospy.get_param('~image_height', 720)
        horizontal_fov = rospy.get_param('~horizontal_fov', 2.0) # radians
        
        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = f_x # Assume square pixels
        c_x = image_width / 2.0
        c_y = image_height / 2.0
        self.camera_matrix = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((5, 1)) # Assume no distortion
        rospy.loginfo(f"Camera Matrix calculated:\n{self.camera_matrix}")

        # --- Camera Extrinsics (Camera to Robot Base) ---
        cam_trans = np.array([0.5, -0.04, 0.57])
        cam_pitch = 0.314 # rpy="0 0.314 0"
        cam_rot_matrix = tfs.euler_matrix(0, cam_pitch, 0, 'sxyz')[:3, :3]
        self.T_robot_camera = np.eye(4)
        self.T_robot_camera[:3, :3] = cam_rot_matrix
        self.T_robot_camera[:3, 3] = cam_trans
        rospy.loginfo(f"Transformation matrix from Robot to Camera:\n{self.T_robot_camera}")

        # --- State Variables ---
        self.latest_odometry = None
        self.bridge = CvBridge()

        # --- ROS Subscribers & Publishers ---
        self.image_sub = rospy.Subscriber('/magv/camera/image_compressed', CompressedImage, self.image_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback, queue_size=1)
        self.aruco_pub = rospy.Publisher('/aruco_info', ArucoInfo, queue_size=10)

        rospy.loginfo("Initialization complete. Waiting for topics...")

    def odometry_callback(self, msg):
        self.latest_odometry = msg

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Failed to decompress image: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        aruco_info_msg = ArucoInfo()
        aruco_info_msg.header = msg.header

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id in enumerate(ids):
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]

                # Create transformation matrix from Camera to Marker
                rot_matrix, _ = cv2.Rodrigues(rvec)
                T_camera_marker = np.eye(4)
                T_camera_marker[:3, :3] = rot_matrix
                T_camera_marker[:3, 3] = tvec

                # Transform marker pose to robot base frame
                T_robot_marker = np.dot(self.T_robot_camera, T_camera_marker)

                # Create ArucoMarker message
                marker_msg = ArucoMarker()
                marker_msg.id = marker_id[0]
                
                trans = tfs.translation_from_matrix(T_robot_marker)
                quat = tfs.quaternion_from_matrix(T_robot_marker)

                marker_msg.pose.position.x = trans[0]
                marker_msg.pose.position.y = trans[1]
                marker_msg.pose.position.z = trans[2]
                marker_msg.pose.orientation.x = quat[0]
                marker_msg.pose.orientation.y = quat[1]
                marker_msg.pose.orientation.z = quat[2]
                marker_msg.pose.orientation.w = quat[3]

                aruco_info_msg.markers.append(marker_msg)
        
        self.aruco_pub.publish(aruco_info_msg)

if __name__ == '__main__':
    try:
        node = ArucoDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass