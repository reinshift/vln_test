#!/usr/bin/env python3
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
        rospy.loginfo("ArUco Detector Node Initializing...")
        rospy.loginfo("ArUco Detector Node Initializing...")

        # --- Parameters ---
        self.marker_size = rospy.get_param('~marker_size', 0.1)  # ArUco码的物理边长 (米)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Historical ArUco detection tracking
        self.historical_detections = {}  # {marker_id: [world_positions]}
        self.detection_threshold = rospy.get_param('~detection_threshold', 0.3)  # meters
        self.max_history_size = rospy.get_param('~max_history_size', 10)

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
        cam_trans_x = rospy.get_param('~camera_translation_x', 0.5)
        cam_trans_y = rospy.get_param('~camera_translation_y', -0.04)
        cam_trans_z = rospy.get_param('~camera_translation_z', 0.57)
        cam_trans = np.array([cam_trans_x, cam_trans_y, cam_trans_z])
        cam_pitch = rospy.get_param('~camera_pitch', 0.314) # rpy="0 0.314 0"
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

        # Subscriber for core node requests to ignore certain ArUco markers
        self.ignore_aruco_sub = rospy.Subscriber('/ignore_aruco', ArucoMarker, self.ignore_aruco_callback, queue_size=10)

        rospy.loginfo("Initialization complete. Waiting for topics...")

    def odometry_callback(self, msg):
        self.latest_odometry = msg

    def ignore_aruco_callback(self, msg):
        """Handle requests to ignore certain ArUco markers"""
        marker_id = msg.id
        world_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # Add to historical detections to prevent future publishing
        if marker_id not in self.historical_detections:
            self.historical_detections[marker_id] = []

        self.historical_detections[marker_id].append(world_pos)

        # Limit history size
        if len(self.historical_detections[marker_id]) > self.max_history_size:
            self.historical_detections[marker_id].pop(0)

        rospy.loginfo(f"Added ArUco marker {marker_id} to ignore list at position {world_pos}")

    def image_callback(self, msg):
        if self.latest_odometry is None:
            rospy.logwarn_throttle(5.0, "No odometry received yet, skipping ArUco detection.")
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Failed to decompress image: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        aruco_info_msg = ArucoInfo()
        aruco_info_msg.header.stamp = self.latest_odometry.header.stamp
        aruco_info_msg.header.frame_id = self.latest_odometry.child_frame_id

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

                # Transform marker pose to world frame using odometry
                if self.latest_odometry is None:
                    rospy.logwarn_throttle(5.0, "No odometry received yet, skipping marker processing.")
                    continue

                odom_pose = self.latest_odometry.pose.pose
                odom_pos = np.array([odom_pose.position.x, odom_pose.position.y, odom_pose.position.z])
                odom_quat = np.array([odom_pose.orientation.x, odom_pose.orientation.y, odom_pose.orientation.z, odom_pose.orientation.w])

                T_world_robot = tfs.quaternion_matrix(odom_quat)
                T_world_robot[:3, 3] = odom_pos

                T_world_marker = np.dot(T_world_robot, T_robot_marker)
                world_pos = tfs.translation_from_matrix(T_world_marker)

                # Check if this marker has been detected before at a similar location
                if not self.is_duplicate_detection(marker_id[0], world_pos):
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

                    # Add to historical detections
                    self.add_to_history(marker_id[0], world_pos)
                else:
                    rospy.logdebug(f"Skipping duplicate detection of ArUco marker {marker_id[0]}")

        self.aruco_pub.publish(aruco_info_msg)

    def is_duplicate_detection(self, marker_id, world_pos):
        """Check if this ArUco marker has been detected before at a similar location"""
        if marker_id not in self.historical_detections:
            return False

        # Check if any previous detection is within threshold distance
        for prev_pos in self.historical_detections[marker_id]:
            distance = np.sqrt(
                (world_pos[0] - prev_pos[0])**2 +
                (world_pos[1] - prev_pos[1])**2 +
                (world_pos[2] - prev_pos[2])**2
            )
            if distance < self.detection_threshold:
                return True

        return False

    def add_to_history(self, marker_id, world_pos):
        """Add a new detection to the historical record"""
        if marker_id not in self.historical_detections:
            self.historical_detections[marker_id] = []

        self.historical_detections[marker_id].append(list(world_pos))

        # Limit history size
        if len(self.historical_detections[marker_id]) > self.max_history_size:
            self.historical_detections[marker_id].pop(0)

        rospy.logdebug(f"Added ArUco marker {marker_id} detection at {world_pos}")

if __name__ == '__main__':
    try:
        rospy.init_node('aruco_detector_node', anonymous=True)
        node = ArucoDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass