#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import math
from collections import deque

# ROS msgs
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from magv_vln_msgs.msg import ArucoInfo, ArucoMarker

try:
    import tf.transformations as tfs
except Exception:
    tfs = None


class ArucoDetectorNode:
    def __init__(self):
        rospy.loginfo("[aruco_detector_node] Initializing...")

        # Parameters
        self.marker_size = rospy.get_param('~marker_size', 1.0)  # meters
        self.marker_id = rospy.get_param('~marker_id', 0)        # only keep this ID
        self.image_width = int(rospy.get_param('~image_width', 1080))
        self.image_height = int(rospy.get_param('~image_height', 720))
        # horizontal_fov in radians per project (e.g., 2.0 ≈ 114.6 deg)
        self.horizontal_fov = float(rospy.get_param('~horizontal_fov', 2.0))

        # Camera extrinsics (camera in base frame)
        cx = float(rospy.get_param('~camera_translation_x', 0.5))
        cy = float(rospy.get_param('~camera_translation_y', -0.04))
        cz = float(rospy.get_param('~camera_translation_z', 0.57))
        self.t_base_cam = np.array([cx, cy, cz], dtype=np.float32)
        self.camera_pitch = float(rospy.get_param('~camera_pitch', 0.314))  # rad, downward

        # Dedup parameters
        self.detection_threshold = float(rospy.get_param('~detection_threshold', 0.3))  # meters
        self.max_history_size = int(rospy.get_param('~max_history_size', 10))

        # Topics
        self.image_topic = rospy.get_param('~image_topic', '/magv/camera/image_compressed/compressed')

        # Publisher
        self.info_pub = rospy.Publisher('/aruco_info', ArucoInfo, queue_size=10)

        # History for dedup (store last positions per id)
        self.history = deque(maxlen=self.max_history_size)

        # Precompute camera intrinsics from FOV
        self.camera_matrix, self.dist_coeffs = self._build_camera_matrix()
        rospy.loginfo("[aruco_detector_node] Camera matrix fx=%.2f, fy=%.2f, cx=%.1f, cy=%.1f" % (
            self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]))

        # Build aruco detector (OpenCV 4.7+ API if available, else fallback)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector_params = self._build_detector_params()
        self.use_new_api = hasattr(cv2.aruco, 'ArucoDetector')
        if self.use_new_api:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            rospy.loginfo("[aruco_detector_node] Using OpenCV ArUco new API")
        else:
            rospy.loginfo("[aruco_detector_node] Using OpenCV ArUco legacy API")

        # Subscriber
        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1)

        rospy.loginfo("[aruco_detector_node] Ready. Subscribing to %s" % self.image_topic)

    def _build_camera_matrix(self):
        # Focal length from horizontal FOV (radians)
        fx = self.image_width / (2.0 * math.tan(self.horizontal_fov / 2.0))
        fy = fx  # assume square pixels
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((5, 1), dtype=np.float32)  # assume no distortion
        return K, dist

    def _build_detector_params(self):
        # Mirror user's working parameters
        if hasattr(cv2.aruco, 'DetectorParameters'):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        # Corner refinement settings (API differs across versions)
        try:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 30
        except Exception:
            pass
        params.minMarkerPerimeterRate = 0.03
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        return params

    @staticmethod
    def _decode_compressed_image(msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img

    def image_callback(self, msg: CompressedImage):
        try:
            image = self._decode_compressed_image(msg)
            if image is None:
                rospy.logwarn_throttle(5.0, "[aruco_detector_node] Failed to decode image")
                return
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.use_new_api:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

            if ids is None or len(ids) == 0:
                return

            ids_flat = ids.flatten()
            # filter only target marker id
            target_indices = np.where(ids_flat == int(self.marker_id))[0]
            if len(target_indices) == 0:
                return

            # Pose estimation for all detected markers (improves stability)
            try:
                rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[aruco_detector_node] estimatePoseSingleMarkers failed: {e}")
                return

            out_markers = []
            for idx in target_indices:
                tvec = np.array(tvecs[idx].reshape(3), dtype=np.float32)  # in camera optical frame (x right, y down, z forward)
                # Transform into base frame
                p_base = self._cam_to_base(tvec)

                # Dedup: skip if too close to last for same id
                if self._is_duplicate(self.marker_id, p_base):
                    continue

                marker = ArucoMarker()
                marker.id = int(self.marker_id)
                pose = Pose()
                pose.position = Point(float(p_base[0]), float(p_base[1]), float(p_base[2]))
                # Orientation not critical for current pipeline; set identity
                if tfs is not None:
                    q = tfs.quaternion_from_euler(0.0, 0.0, 0.0)
                else:
                    q = (0.0, 0.0, 0.0, 1.0)
                pose.orientation = Quaternion(*q)
                marker.pose = pose
                out_markers.append(marker)
                # Update history
                self.history.append((int(self.marker_id), float(p_base[0]), float(p_base[1]), float(p_base[2])))

            if out_markers:
                info = ArucoInfo()
                info.header = Header(stamp=rospy.Time.now(), frame_id='base_footprint')
                info.markers = out_markers
                self.info_pub.publish(info)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[aruco_detector_node] image_callback exception: {e}")

    def _cam_to_base(self, p_cam: np.ndarray) -> np.ndarray:
        # Map OpenCV camera optical coords (x right, y down, z forward)
        # to base frame (x forward, y left, z up) with pitch around base Y
        # First nominal mapping when pitch=0
        px = float(p_cam[0])
        py = float(p_cam[1])
        pz = float(p_cam[2])
        p_nominal = np.array([pz, -px, -py], dtype=np.float32)  # [x_b, y_b, z_b] without pitch
        # Apply pitch rotation around base Y (downward pitch positive)
        c = math.cos(self.camera_pitch)
        s = math.sin(self.camera_pitch)
        R_y = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float32)
        p_rot = R_y.dot(p_nominal)
        # Add camera translation in base frame
        p_base = self.t_base_cam + p_rot
        return p_base

    def _is_duplicate(self, marker_id: int, p_base: np.ndarray) -> bool:
        # Check last records of the same id within threshold
        th = self.detection_threshold
        for mid, x, y, z in reversed(self.history):
            if mid != marker_id:
                continue
            dx = float(p_base[0]) - x
            dy = float(p_base[1]) - y
            dz = float(p_base[2]) - z
            if math.sqrt(dx*dx + dy*dy + dz*dz) <= th:
                return True
            # only compare with the most recent same-id record
            break
        return False


def main():
    rospy.init_node('aruco_detector_node', anonymous=False)
    node = ArucoDetectorNode()
    rospy.spin()


if __name__ == '__main__':
    main()

