#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf.transformations as tfs
from PIL import Image
import io

# ROS Messages
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

# Custom Messages
from magv_vln_msgs.msg import ArucoInfo, ArucoMarker

class ArucoDetectorNode:
    def __init__(self):
        rospy.loginfo("ArUco Detector Node Initializing...")

        # --- Parameters ---
        self.marker_size = rospy.get_param('~marker_size', 1.0)
        self.aruco_dict_name = rospy.get_param('~aruco_dictionary', 'DICT_4X4_50')

        # Check if OpenCV has aruco module
        self.aruco_available = hasattr(cv2, 'aruco')
        self.aruco_dict = None
        self.active_dict_name = None
        self._dict_objs = []

        if not self.aruco_available:
            rospy.logerr("OpenCV was built without the 'aruco' module. ArUco detection disabled; will publish empty /aruco_info heartbeats only.")
        else:
            # Resolve primary dictionary id safely
            try:
                dict_id = getattr(cv2.aruco, self.aruco_dict_name)
            except Exception:
                rospy.logwarn(f"Unknown aruco dictionary '{self.aruco_dict_name}', fallback to DICT_4X4_50")
                self.aruco_dict_name = 'DICT_4X4_50'
                try:
                    dict_id = getattr(cv2.aruco, self.aruco_dict_name)
                except Exception:
                    dict_id = None
            try:
                if dict_id is not None:
                    self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            except Exception as e:
                rospy.logwarn(f"Failed to getPredefinedDictionary for {self.aruco_dict_name}: {e}")
                self.aruco_dict = None

            # Detector parameters (robust defaults)
            try:
                self.aruco_params = cv2.aruco.DetectorParameters()
            except Exception:
                # 兼容旧 API
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            for name, val in [
                ("adaptiveThreshWinSizeMin", 3),
                ("adaptiveThreshWinSizeMax", 23),
                ("adaptiveThreshWinSizeStep", 10),
                ("cornerRefinementWinSize", 5),
                ("cornerRefinementMaxIterations", 30),
                ("minMarkerPerimeterRate", 0.03),
                ("maxMarkerPerimeterRate", 4.0),
                ("polygonalApproxAccuracyRate", 0.05),
            ]:
                try:
                    setattr(self.aruco_params, name, val)
                except Exception:
                    pass
            try:
                self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            except Exception:
                pass

            # Try multiple dictionaries like the test script; primary from ~aruco_dictionary
            self.try_multiple_dicts = bool(rospy.get_param('~try_multiple_dictionaries', True))
            default_dicts = [
                'DICT_4X4_50','DICT_4X4_100','DICT_4X4_250','DICT_4X4_1000'
            ]
            self.dict_candidates = rospy.get_param('~aruco_dictionaries', default_dicts)
            # Ensure primary dict name is first
            if self.aruco_dict_name in self.dict_candidates:
                try:
                    self.dict_candidates.remove(self.aruco_dict_name)
                except ValueError:
                    pass
            self.dict_candidates.insert(0, self.aruco_dict_name)
            # Build dict objects list
            for name in self.dict_candidates:
                try:
                    did = getattr(cv2.aruco, name)
                    d = cv2.aruco.getPredefinedDictionary(did)
                    self._dict_objs.append((name, d))
                except Exception:
                    rospy.logwarn_throttle(10.0, f"Aruco dictionary not available in OpenCV: {name}")


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

        # --- ROS Subscribers & Publishers ---
        # 订阅图像话题（可参数化），默认 '/magv/camera/image_compressed/compressed'
        self.image_topic = rospy.get_param('~image_topic', '/magv/camera/image_compressed/compressed')
        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback, queue_size=1)
        self.aruco_pub = rospy.Publisher('/aruco_info', ArucoInfo, queue_size=10, latch=True)

        # Subscriber for core node requests to ignore certain ArUco markers
        self.ignore_aruco_sub = rospy.Subscriber('/ignore_aruco', ArucoMarker, self.ignore_aruco_callback, queue_size=10)

        # Heartbeat/diagnostics: periodically publish empty ArucoInfo so /aruco_info is visible
        self.publish_heartbeat = rospy.get_param('~publish_heartbeat', True)
        self.heartbeat_rate_hz = float(rospy.get_param('~heartbeat_rate_hz', 1.0))
        self.last_image_time = 0.0
        if self.publish_heartbeat and self.heartbeat_rate_hz > 0:
            self._hb_timer = rospy.Timer(rospy.Duration(1.0 / max(0.1, self.heartbeat_rate_hz)), self._heartbeat)

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
        """Handle incoming compressed images, detect ArUco markers, and publish poses.
        - Works even if odometry is not yet available (publishes in robot base frame).
        - Uses cv_bridge to decode JPEG to BGR ndarray; falls back to PIL if needed.
        """
        # Decode compressed image to OpenCV BGR ndarray
        try:
            import numpy as _np
            cv_bgr = None
            try:
                cv_bgr = cv2.imdecode(_np.frombuffer(msg.data, _np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                cv_bgr = None
            if cv_bgr is None:
                raise RuntimeError('cv2.imdecode failed')
        except Exception as e1:
            try:
                pil_img = Image.open(io.BytesIO(msg.data)).convert('RGB')
                cv_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e2:
                rospy.logerr(f"Failed to decode compressed image: cv_bridge error={e1}; PIL fallback error={e2}")
                # publish empty info so topic stays alive for diagnosis
                empty = ArucoInfo()
                empty.header.stamp = getattr(msg, 'header', None).stamp if hasattr(msg, 'header') else rospy.Time.now()
                empty.header.frame_id = getattr(self, 'base_frame_id', 'base_footprint')
                self.aruco_pub.publish(empty)
                return

        gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids = None, None
        if not getattr(self, 'aruco_available', False):
            # Without aruco module, just publish empty message
            empty = ArucoInfo()
            empty.header.stamp = aruco_info_msg.header.stamp
            empty.header.frame_id = aruco_info_msg.header.frame_id
            self.aruco_pub.publish(empty)
            return
        # Try multiple dictionaries if enabled; remember the one that works
        if self.try_multiple_dicts:
            for name, d in self._dict_objs:
                try:
                    # New API: ArucoDetector
                    try:
                        detector = cv2.aruco.ArucoDetector(d, self.aruco_params)
                        c, i, _ = detector.detectMarkers(gray)
                    except Exception:
                        # Fallback API
                        c, i, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.aruco_params)
                    if i is not None and len(i) > 0:
                        corners, ids = c, i
                        self.active_dict_name = name
                        rospy.loginfo_throttle(5.0, f"ArUco detected with dictionary: {name} (ids={i.flatten().tolist()})")
                        break
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"detectMarkers failed with dict {name}: {e}")
        # If none found via multi-try, fall back to configured dict
        if corners is None:
            try:
                c, i, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                if i is not None and len(i) > 0:
                    corners, ids = c, i
                    self.active_dict_name = self.aruco_dict_name
            except Exception as e:
                rospy.logerr_throttle(5.0, f"detectMarkers with default dict failed: {e}")

        # Prepare output message
        aruco_info_msg = ArucoInfo()
        # Prefer image timestamp; fall back to odom timestamp if available
        stamp = getattr(msg, 'header', None).stamp if hasattr(msg, 'header') else None
        if stamp is None and self.latest_odometry is not None:
            stamp = self.latest_odometry.header.stamp
        aruco_info_msg.header.stamp = stamp if stamp is not None else rospy.Time.now()

        # Frame: use odom's child frame if available; otherwise fallback to a base frame param
        self.base_frame_id = getattr(self, 'base_frame_id', rospy.get_param('~base_frame_id', 'base_footprint'))
        if self.latest_odometry is not None and self.latest_odometry.child_frame_id:
            aruco_info_msg.header.frame_id = self.latest_odometry.child_frame_id
        else:
            aruco_info_msg.header.frame_id = self.base_frame_id

        if ids is None or len(ids) == 0:
            rospy.loginfo_throttle(5.0, f"No ArUco detected (dict={self.aruco_dict_name}, size={self.marker_size}m, topic={self.image_topic}).")
            self.aruco_pub.publish(aruco_info_msg)
            return

        # Estimate marker poses
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
        except Exception as e:
            rospy.logerr(f"estimatePoseSingleMarkers failed: {e}")
            self.aruco_pub.publish(aruco_info_msg)
            return

        # If odom available, precompute world transform once
        have_odom = self.latest_odometry is not None
        if have_odom:
            odom_pose = self.latest_odometry.pose.pose
            odom_pos = np.array([odom_pose.position.x, odom_pose.position.y, odom_pose.position.z])
            odom_quat = np.array([odom_pose.orientation.x, odom_pose.orientation.y, odom_pose.orientation.z, odom_pose.orientation.w])
            T_world_robot = tfs.quaternion_matrix(odom_quat)
            T_world_robot[:3, 3] = odom_pos

        for i, id_arr in enumerate(ids):
            marker_id = int(id_arr[0]) if hasattr(id_arr, '__iter__') else int(id_arr)
            tvec = tvecs[i][0]
            rvec = rvecs[i][0]

            # Camera->Marker transform
            rot_matrix, _ = cv2.Rodrigues(rvec)
            T_camera_marker = np.eye(4)
            T_camera_marker[:3, :3] = rot_matrix
            T_camera_marker[:3, 3] = tvec

            # Robot(base)->Marker transform
            T_robot_marker = np.dot(self.T_robot_camera, T_camera_marker)

            # Prepare message in robot base frame (independent of odom)
            trans_rb = tfs.translation_from_matrix(T_robot_marker)
            quat_rb = tfs.quaternion_from_matrix(T_robot_marker)

            marker_msg = ArucoMarker()
            marker_msg.id = marker_id
            marker_msg.pose.position.x = float(trans_rb[0])
            marker_msg.pose.position.y = float(trans_rb[1])
            marker_msg.pose.position.z = float(trans_rb[2])
            marker_msg.pose.orientation.x = float(quat_rb[0])
            marker_msg.pose.orientation.y = float(quat_rb[1])
            marker_msg.pose.orientation.z = float(quat_rb[2])
            marker_msg.pose.orientation.w = float(quat_rb[3])

            # Deduplication using world pose if odom is available
            if have_odom:
                T_world_marker = np.dot(T_world_robot, T_robot_marker)
                world_pos = tfs.translation_from_matrix(T_world_marker)
                if not self.is_duplicate_detection(marker_id, world_pos):
                    aruco_info_msg.markers.append(marker_msg)
                    self.add_to_history(marker_id, world_pos)
                else:
                    rospy.logdebug(f"Skipping duplicate detection of ArUco marker {marker_id}")
            else:
                # Without odom, do not apply history-based filtering
                aruco_info_msg.markers.append(marker_msg)

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

    def _heartbeat(self, event):
        try:
            if (rospy.Time.now().to_sec() - self.last_image_time) > 1.0:
                msg = ArucoInfo()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = getattr(self, 'base_frame_id', 'base_footprint')
                self.aruco_pub.publish(msg)
        except Exception:
            pass


if __name__ == '__main__':
    try:
        rospy.init_node('aruco_detector_node', anonymous=True)
        node = ArucoDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass