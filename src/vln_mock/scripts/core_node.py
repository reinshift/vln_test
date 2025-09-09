#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import rospy
import json
import numpy as np
import cv2
import math
import re
from threading import Lock
from collections import deque
import tf.transformations as tfs

# ROS Messages
from std_msgs.msg import String, Int32, Bool, Float32MultiArray, MultiArrayDimension
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, Twist, PoseStamped, Quaternion
from sensor_msgs.msg import CompressedImage

# Custom Messages
from magv_vln_msgs.msg import (
    VehicleStatus, ValueMap, PathPoint, PositionCommand,
    DetectedObjectArray, Detection2DArray
)
from magv_vln_msgs.msg import ArucoInfo, ArucoMarker

class CoreNode:
    def __init__(self):
        rospy.loginfo("Core Node initializing...")

        # State variables
        self.current_state = VehicleStatus.STATE_IDLE
        self.current_subtasks = []
        self.current_subtask_index = 0
        self.current_subtask = None
        # Track last seen subtasks/index to avoid resetting during steady NAVIGATION updates
        self.last_subtasks_json = None
        self.last_subtask_index = None
        self.occupancy_grid = None
        self.value_map = None
        self.current_path_points = []
        self.current_path_index = 0
        self.aruco_detections = []
        self.detected_objects = {} # Stores positions of detected objects
        self.directional_detections = {} # Stores {label: [(yaw, detection_msg)]}
        self.navigation_active = False
        self.scan_and_plan_complete = False
        self.task_completed = False
        self.scan_in_progress = False
        self.scan_start_time = None
        self.scan_start_yaw = 0.0
        self.scan_last_yaw = 0.0
        self.scan_accum_angle = 0.0
        self.scan_ctrl_timer = None
        self.current_yaw = 0.0
        self.current_pose = None
        self.last_odom_time = None

        # ArUco target tracking (direct navigation mode)
        self.aruco_target_active = False
        self.aruco_target_id = None
        self.aruco_last_pose = None  # geometry_msgs/Pose in body frame
        self.aruco_close_counter = 0
        self.aruco_arrival_distance = rospy.get_param('~aruco_arrival_distance', 0.3)

        # Thread safety
        self.state_lock = Lock()

        # Publishers
        self.value_map_pub = rospy.Publisher('/value_map', ValueMap, queue_size=1)
        self.value_map_preview_pub = rospy.Publisher('/value_map_preview', Float32MultiArray, queue_size=1)
        self.path_point_pub = rospy.Publisher('/path_point', PathPoint, queue_size=10)
        self.controller_discrete_pub = rospy.Publisher('/world_goal', PositionCommand, queue_size=10)
        self.controller_body_pub = rospy.Publisher('/body_goal', PositionCommand, queue_size=10)
        # Prefer sending velocity goals to the controller so it can enforce limits
        self.controller_continuous_pub = rospy.Publisher('/magv/omni_drive_controller/cmd_vel', Twist, queue_size=10)
        self.controller_velocity_pub = rospy.Publisher('/velocity_goal', Twist, queue_size=10)
        self.status_feedback_pub = rospy.Publisher('/core_feedback', String, queue_size=10)
        self.vlm_query_pub = rospy.Publisher('/vlm_query', String, queue_size=10)
        self.final_status_pub = rospy.Publisher('/status', Int32, queue_size=10)
        self.dino_prompt_pub = rospy.Publisher('/grounding_dino/prompt', String, queue_size=10)

        # Subscribers
        self.vln_status_sub = rospy.Subscriber('/vln_status', VehicleStatus, self.vln_status_callback, queue_size=1)
        self.occupancy_grid_sub = rospy.Subscriber('/occupancy_grid', OccupancyGrid, self.occupancy_grid_callback, queue_size=1)
        self.aruco_info_sub = rospy.Subscriber('/aruco_info', ArucoInfo, self.aruco_info_callback, queue_size=1)
        # Subscribe to the correct compressed image topic from rosbag
        self.image_sub = rospy.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage, self.image_callback, queue_size=1)
        self.vlm_response_sub = rospy.Subscriber('/vlm_response', String, self.vlm_response_callback, queue_size=1)
        self.odometry_sub = rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback, queue_size=1)
        self.dino_detections_sub = rospy.Subscriber('/grounding_dino/detections', Detection2DArray, self.dino_detections_callback, queue_size=1)


        # Parameters
        self.grid_resolution = rospy.get_param('~grid_resolution', 0.1)  # meters per cell
        self.path_planning_distance = rospy.get_param('~path_planning_distance', 2.0)  # meters
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.5)  # meters
        self.aruco_detection_threshold = rospy.get_param('~aruco_detection_threshold', 0.3)  # meters
        # Use same default caps as controller for consistency
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 6.28)
        # Scan duration (seconds)
        self.scan_duration_sec = rospy.get_param('~scan_duration_sec', 3.0)
        # Scan yaw alignment tolerance (radians)
        self.scan_yaw_tolerance = rospy.get_param('~scan_yaw_tolerance', 0.05)
        # Progress controller gain to match 2π within duration
        self.scan_progress_kp = rospy.get_param('~scan_progress_kp', 2.0)
        # Safety timeout factor: stop if scan exceeds duration * factor
        self.scan_timeout_factor = rospy.get_param('~scan_timeout_factor', 1.7)
        # Odometry freshness threshold (seconds)
        self.odom_stale_threshold = rospy.get_param('~odom_stale_threshold', 0.5)

        # Camera parameters for projecting detections
        self.image_width = rospy.get_param('~image_width', 1080)
        self.horizontal_fov = rospy.get_param('~horizontal_fov', 2.0) # radians
        self.f_x = (self.image_width / 2.0) / np.tan(self.horizontal_fov / 2.0)

        # Value map smoothing (softmax-like) temperature; 0 disables smoothing
        try:
            self.value_smoothing_tau = float(rospy.get_param('~value_smoothing_tau', 0.0))
        except Exception:
            self.value_smoothing_tau = 0.0
        if self.value_smoothing_tau is not None and self.value_smoothing_tau < 0.0:
            self.value_smoothing_tau = 0.0

        # Directional narrow band parameters (for fallback directional guidance)
        self.dir_band_half_width_m = rospy.get_param('~directional_band_half_width_m', 0.4)
        self.dir_longitudinal_gain = rospy.get_param('~directional_longitudinal_gain', 10.0)
        self.dir_lateral_sigma_m = rospy.get_param('~directional_lateral_sigma_m', self.dir_band_half_width_m / 2.0)

        # Near-obstacle attractor in directional band
        self.near_obstacle_boost_enable = rospy.get_param('~near_obstacle_boost_enable', True)
        self.near_obstacle_boost_gain = rospy.get_param('~near_obstacle_boost_gain', 300.0)
        self.near_obstacle_lateral_min_weight = rospy.get_param('~near_obstacle_lateral_min_weight', 0.1)
        self.near_obstacle_step_m = rospy.get_param('~near_obstacle_step_m', self.grid_resolution)


        # Control timer for navigation
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_timer_callback)

        rospy.loginfo("Core Node initialized successfully")

    def get_yaw_from_quaternion(self, quaternion):
        """Extract yaw angle from quaternion"""
        euler = tfs.euler_from_quaternion([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        ])
        return euler[2]  # yaw

    def vln_status_callback(self, msg):
        """Handle VLN status updates from state machine"""
        with self.state_lock:
            self.current_state = msg.state

            # Parse subtasks if available
            if msg.current_subtask_json:
                try:
                    incoming_json = msg.current_subtask_json
                    incoming_index = msg.current_subtask_index
                    # Only refresh/clear when subtasks changed or index advanced
                    should_reset = (self.last_subtasks_json != incoming_json) or (self.last_subtask_index != incoming_index)

                    if should_reset:
                        self.current_subtasks = json.loads(incoming_json)
                        self.current_subtask_index = incoming_index
                        if 0 <= self.current_subtask_index < len(self.current_subtasks):
                            self.current_subtask = self.current_subtasks[self.current_subtask_index]
                        else:
                            self.current_subtask = None
                        # Reset scan and plan state only when new subtasks/index detected
                        self.scan_and_plan_complete = False
                        self.directional_detections = {}  # Clear previous detections
                        self.current_path_points = []  # Clear previous path points
                        self.current_path_index = 0
                        self.last_subtasks_json = incoming_json
                        self.last_subtask_index = incoming_index
                        rospy.loginfo(f"Updated subtasks: {len(self.current_subtasks)} tasks, current index: {self.current_subtask_index}")
                        rospy.loginfo(f"Current subtask: {self.current_subtask}")
                except json.JSONDecodeError as e:
                    rospy.logerr(f"Failed to parse subtasks JSON: {e}")
                    return

            # Handle state transitions
            if msg.state == VehicleStatus.STATE_INITIALIZING:
                self.handle_initialization_state()
            elif msg.state == VehicleStatus.STATE_NAVIGATION:
                self.handle_navigation_state()
            elif msg.state == VehicleStatus.STATE_EMERGENCY_STOP:
                self.handle_emergency_stop()

    def handle_initialization_state(self):
        """
        Handle initialization state.
        This now involves a "scan and plan" phase where the robot rotates
        to build a map of its surroundings using GroundingDINO before planning.
        """
        if self.scan_and_plan_complete:
            rospy.loginfo("Scan and plan already complete for this subtask.")
            # Notify state machine again in case the first message was missed
            feedback_msg = String()
            feedback_msg.data = json.dumps({
                "status": "initialization_complete",
                "path_points_count": len(self.current_path_points)
            })
            self.status_feedback_pub.publish(feedback_msg)
            return

        rospy.loginfo("Starting scan and plan phase...")
        rospy.loginfo(f"Current subtasks: {self.current_subtasks}")
        rospy.loginfo(f"Current subtask index: {self.current_subtask_index}")
        rospy.loginfo(f"Current subtask: {self.current_subtask}")

        if not self.current_subtask:
            rospy.logerr("No current subtask available for initialization!")
            rospy.logerr(f"Subtasks list: {self.current_subtasks}")
            rospy.logerr(f"Subtask index: {self.current_subtask_index}")
            # Try to proceed with a default subtask to avoid getting stuck
            if self.current_subtasks and len(self.current_subtasks) > 0:
                rospy.logwarn("Attempting to use first available subtask...")
                self.current_subtask = self.current_subtasks[0]
                self.current_subtask_index = 0
            else:
                rospy.logerr("No subtasks available at all! Cannot proceed.")
                return

        # 1. Publish the goal object as a prompt for GroundingDINO
        goal = self.current_subtask.get('goal', None)
        if goal and goal != 'null':
            prompt_msg = String()
            prompt_msg.data = goal
            self.dino_prompt_pub.publish(prompt_msg)
            rospy.loginfo(f"Published GroundingDINO prompt: '{goal}'")
        else:
            rospy.loginfo("No specific goal in subtask, will rely on directional guidance.")

        # 2. Command the robot to perform a 360-degree scan (only once per subtask)
        if self.scan_in_progress:
            rospy.logdebug("Scan already in progress; skip re-triggering rotation.")
            return
        self.scan_in_progress = True

        rospy.loginfo("Commanding 360-degree rotation for scanning...")
        scan_cmd = Twist()
        scan_cmd.angular.z = self.max_angular_vel
        try:
            self.controller_velocity_pub.publish(scan_cmd)
        except Exception:
            self.controller_continuous_pub.publish(scan_cmd)

        # 3. Start closed-loop scan control to complete 2π within duration and realign yaw
        self.scan_start_time = rospy.Time.now()
        self.scan_start_yaw = self.current_yaw
        self.scan_last_yaw = self.current_yaw
        self.scan_accum_angle = 0.0
        # Create/update control timer (50 Hz)
        if self.scan_ctrl_timer is not None:
            try:
                self.scan_ctrl_timer.shutdown()
            except Exception:
                pass
        self.scan_ctrl_timer = rospy.Timer(rospy.Duration(0.02), self.scan_control_loop)

    def finish_scan_and_plan(self, event):
        """
        Called by a timer after the robot has finished its 360-degree scan.
        Stops the robot, computes the map, generates a path, and notifies the state machine.
        """
        rospy.loginfo("Rotation finished. Stopping robot and starting planning.")
        # Mark scan phase ended
        self.scan_in_progress = False
        # Stop scan control timer if running
        if self.scan_ctrl_timer is not None:
            try:
                self.scan_ctrl_timer.shutdown()
            except Exception:
                pass
            self.scan_ctrl_timer = None

        # 1. Stop the robot's rotation
        stop_cmd = Twist()
        try:
            self.controller_velocity_pub.publish(stop_cmd)
        except Exception:
            self.controller_continuous_pub.publish(stop_cmd)

        # Clear the GroundingDINO prompt
        self.dino_prompt_pub.publish(String(data=""))

        if self.occupancy_grid is None:
            rospy.logwarn("Occupancy grid not available yet; will retry planning shortly without re-rotating.")
            rospy.Timer(rospy.Duration(1.0), self.try_plan_after_grid, oneshot=True)
            return

        # Proceed to planning now that grid is available
        self._do_plan_and_notify()

    def scan_control_loop(self, event):
        """Closed-loop control to achieve 2π rotation within scan_duration_sec and realign to start yaw."""
        if not self.scan_in_progress:
            return
        now = rospy.Time.now()
        elapsed = (now - self.scan_start_time).to_sec() if self.scan_start_time else 0.0

        # Safety timeout
        if elapsed > self.scan_duration_sec * self.scan_timeout_factor:
            rospy.logwarn("Scan exceeded safety timeout, forcing stop and planning.")
            self.finish_scan_and_plan(None)
            return

        # Determine odometry freshness; if stale, drive purely by time profile and stop at duration
        odom_stale = (self.last_odom_time is None) or ((now - self.last_odom_time).to_sec() > self.odom_stale_threshold)

        # Update current yaw and accumulated angle
        current_yaw = self.current_yaw
        if not odom_stale:
            dyaw = self.normalize_angle(current_yaw - self.scan_last_yaw)
            self.scan_accum_angle += abs(dyaw)
            self.scan_last_yaw = current_yaw

        two_pi = 2.0 * math.pi
        # Desired progress over time
        desired_progress = max(0.0, min(two_pi, two_pi * (elapsed / max(1e-3, self.scan_duration_sec))))
        progress_error = desired_progress - self.scan_accum_angle

        cmd = Twist()

        if elapsed < self.scan_duration_sec:
            # Feedforward + proportional on progress to match schedule
            omega_ff = two_pi / max(1e-3, self.scan_duration_sec)
            if odom_stale:
                # Without odom, just use feedforward profile
                omega_cmd = omega_ff
            else:
                omega_cmd = omega_ff + self.scan_progress_kp * progress_error
            # Clamp
            omega_cmd = max(-self.max_angular_vel, min(self.max_angular_vel, omega_cmd))
            # Ensure positive rotation direction (counter-clockwise), but allow correction to stay on schedule
            if omega_cmd < 0.2:
                omega_cmd = 0.2
            cmd.angular.z = omega_cmd
            try:
                self.controller_velocity_pub.publish(cmd)
            except Exception:
                self.controller_continuous_pub.publish(cmd)
            return
        else:
            # If no odom, end scan exactly at duration without alignment
            if odom_stale:
                self.finish_scan_and_plan(None)
                return
            # Alignment phase: bring yaw back to start yaw within tolerance
            yaw_err = self.normalize_angle(self.scan_start_yaw - current_yaw)
            if abs(yaw_err) <= self.scan_yaw_tolerance:
                self.finish_scan_and_plan(None)
                return
            # Proportional correction with clamp and minimum speed
            omega_c = self.scan_progress_kp * yaw_err
            max_omega = min(self.max_angular_vel, 1.5)  # limit during alignment
            min_omega = 0.15
            if omega_c >= 0:
                omega_c = max(min_omega, min(max_omega, omega_c))
            else:
                omega_c = -max(min_omega, min(max_omega, -omega_c))
            cmd.angular.z = omega_c
            try:
                self.controller_velocity_pub.publish(cmd)
            except Exception:
                self.controller_continuous_pub.publish(cmd)

    def try_plan_after_grid(self, event):
        """Retry planning until occupancy grid arrives; do not restart rotation."""
        if self.occupancy_grid is None:
            rospy.logwarn_throttle(5.0, "Waiting for occupancy grid to plan...")
            rospy.Timer(rospy.Duration(1.0), self.try_plan_after_grid, oneshot=True)
            return
        self._do_plan_and_notify()

    def _do_plan_and_notify(self):
        """Compute value map, generate path, mark complete, and notify state machine."""
        # 2. Compute the value map using the data gathered during the scan
        self.compute_value_map()

        # 3. Generate path points based on the new value map
        self.generate_path_points()

        # 4. Mark this phase as complete and notify the state machine
        self.scan_and_plan_complete = True
        feedback_msg = String()
        feedback_msg.data = json.dumps({
            "status": "initialization_complete",
            "path_points_count": len(self.current_path_points)
        })
        self.status_feedback_pub.publish(feedback_msg)
        rospy.loginfo("Scan and plan phase complete. Notified state machine.")

    def odometry_callback(self, msg):
        """Update the robot's current yaw and pose from odometry data."""
        self.current_pose = msg.pose.pose
        orientation_q = msg.pose.pose.orientation
        euler = tfs.euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])
        self.current_yaw = euler[2]
        self.last_odom_time = rospy.Time.now()

    def dino_detections_callback(self, msg):
        """
        Store detections from GroundingDINO, associating them with the current yaw.
        This is crucial for the rotational scan.
        """
        # Only store detections if we are in the initialization (scanning) phase
        if self.current_state != VehicleStatus.STATE_INITIALIZING:
            return

        for detection in msg.detections:
            label = (detection.label or '').strip().lower()
            if label not in self.directional_detections:
                self.directional_detections[label] = []
            # Store the yaw and the full detection message
            self.directional_detections[label].append((self.current_yaw, detection))
            rospy.logdebug(f"Stored detection for '{label}' at yaw {self.current_yaw:.2f}")



    def vlm_response_callback(self, msg):
        """Handle VLM response"""
        rospy.loginfo("Received VLM response")
        self.handle_vlm_response(msg.data)

    def handle_navigation_state(self):
        """Handle navigation state - execute path following"""
        rospy.loginfo("Handling navigation state...")
        self.navigation_active = True

    def handle_emergency_stop(self):
        """Handle emergency stop - stop all motion"""
        rospy.logwarn("Emergency stop activated!")
        self.navigation_active = False

        # Send stop command
        stop_cmd = Twist()
        self.controller_continuous_pub.publish(stop_cmd)

    def occupancy_grid_callback(self, msg):
        """Handle occupancy grid updates"""
        self.occupancy_grid = msg
        rospy.loginfo("Received occupancy grid update")

    def aruco_info_callback(self, msg):
        """Handle ArUco detection updates"""
        if msg.markers:
            self.aruco_detections = msg.markers
            rospy.loginfo(f"Detected {len(msg.markers)} ArUco markers")

            # If we're navigating and detect ArUco markers, handle them
            if self.navigation_active:
                self.handle_aruco_detection()


    def image_callback(self, msg):
        """Handle camera image updates (for VLM queries)"""
        # Store latest image for VLM queries
        self.latest_image = msg



    def compute_value_map(self):
        """Compute value map based on current subtask and occupancy grid"""
        if not self.occupancy_grid or not self.current_subtask:
            return

        rospy.loginfo(f"Computing value map for subtask: {self.current_subtask}")

        # Create value map with same dimensions as occupancy grid
        value_map = ValueMap()
        value_map.header = self.occupancy_grid.header
        value_map.info = self.occupancy_grid.info

        width = self.occupancy_grid.info.width
        height = self.occupancy_grid.info.height

        # Initialize value data
        values = np.zeros(width * height, dtype=np.float32)

        # Get subtask direction and goal
        # Extract direction from subtask, which could be under keys like 'subtask_1', 'subtask_2', etc.
        direction = 'forward' # Default value
        for key, value in self.current_subtask.items():
            if 'subtask' in key:
                direction = (value or '').strip().lower()
                break
        goal = self.current_subtask.get('goal', None)
        if isinstance(goal, str):
            goal = goal.strip().lower() if goal else None

        # Get current robot yaw for orientation-aware value calculation
        current_yaw = 0.0
        if self.current_pose:
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Compute values based on direction and goal
        for y in range(height):
            for x in range(width):
                idx = y * width + x

                # Skip occupied cells (set very low value)
                if self.occupancy_grid.data[idx] > 50:  # Occupied
                    values[idx] = -1000.0
                    continue

                # Convert grid coordinates to world coordinates
                world_x = self.occupancy_grid.info.origin.position.x + x * self.occupancy_grid.info.resolution
                world_y = self.occupancy_grid.info.origin.position.y + y * self.occupancy_grid.info.resolution

                # Compute value based on direction preference
                value = self.compute_cell_value(world_x, world_y, direction, goal, current_yaw)
                values[idx] = value

        # Optional value smoothing (softmax-like) excluding obstacles
        tau = getattr(self, 'value_smoothing_tau', 0.0) or 0.0
        if tau > 1e-6:
            try:
                # mask: True for free cells
                occ = np.array(self.occupancy_grid.data, dtype=np.int16).reshape((height, width))
                free_mask = (occ <= 50)
                vals2d = values.reshape((height, width))
                # For numerical stability: subtract max over free cells
                max_free = np.max(vals2d[free_mask]) if np.any(free_mask) else 0.0
                exp_vals = np.zeros_like(vals2d, dtype=np.float32)
                exp_vals[free_mask] = np.exp((vals2d[free_mask] - max_free) / tau)
                # Preserve obstacles as -1000, scale free cells to [0,1] by normalizing softmax weights
                sum_exp = np.sum(exp_vals[free_mask])
                if sum_exp > 0:
                    smoothed = np.full_like(vals2d, -1000.0, dtype=np.float32)
                    # Convert weights to a value field in [0,1] then rescale back to original dynamic range around max
                    weights = exp_vals[free_mask] / sum_exp
                    # Map weights to a comparable scale: use (weights * N) as relative scores (N=free cells count)
                    N = np.count_nonzero(free_mask)
                    rel = weights * max(1, N)
                    smoothed_vals = rel.astype(np.float32)
                    smoothed[free_mask] = smoothed_vals
                    values = smoothed.ravel()
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Value smoothing failed: {e}")

        value_map.data = values.tolist()
        self.value_map = value_map
        self.value_map_pub.publish(value_map)

        rospy.loginfo("Value map computed and published")

        # Also publish a 20x20 preview matrix for quick visualization
        try:
            # Reshape to 2D (height, width) for preview sampling
            values_2d = values.reshape((height, width))
            preview = self._build_value_map_preview(values_2d, width, height, target_size=20)
            self.value_map_preview_pub.publish(preview)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to publish value map preview: {e}")

    def _build_value_map_preview(self, values_np, width, height, target_size=20):
        """
        Build a Float32MultiArray preview of size target_size x target_size from the full value map.
        Uses nearest-neighbor sampling for robustness and speed.
        """
        # values_np is a 2D array shaped (height, width)
        ys = np.linspace(0, height - 1, target_size).astype(int)
        xs = np.linspace(0, width - 1, target_size).astype(int)
        sampled = values_np[ys[:, None], xs[None, :]]  # shape (target_size, target_size)

        msg = Float32MultiArray()
        # Layout: 2D (rows=target_size, cols=target_size), row-major
        dim_rows = MultiArrayDimension(label='rows', size=target_size, stride=target_size * target_size)
        dim_cols = MultiArrayDimension(label='cols', size=target_size, stride=target_size)
        msg.layout.dim = [dim_rows, dim_cols]
        msg.layout.data_offset = 0
        msg.data = sampled.astype(np.float32).ravel().tolist()
        return msg

    def compute_cell_value(self, x, y, direction, goal, current_yaw):
        """
        Compute value for a single cell. This now prioritizes directional detections
        from the rotational scan, creating high-value sectors in the direction
        where the goal object was seen.
        """
        base_value = 0.0
        goal_seen = False

        # --- Directional Value from Rotational Scan ---
        if goal and goal in self.directional_detections and self.current_pose is not None:
            # Compute angle of the cell relative to robot position (world -> robot-centered)
            rx = x - self.current_pose.position.x
            ry = y - self.current_pose.position.y
            cell_angle = math.atan2(ry, rx)

            for (detection_yaw, detection) in self.directional_detections[goal]:
                # Calculate the angular cone where the object was detected
                bbox = detection.bbox
                center_pixel = bbox.center.x
                width_pixel = bbox.size_x

                # Angle of the bbox center relative to the camera's center view
                center_angle_offset = math.atan((self.image_width / 2 - center_pixel) / self.f_x)
                half_width_angle = math.atan((width_pixel / 2) / self.f_x)

                # The absolute angle in the world frame where the object was detected
                absolute_detection_angle = self.normalize_angle(detection_yaw + center_angle_offset)
                angle_diff = self.normalize_angle(cell_angle - absolute_detection_angle)

                # Check if the cell's angle is within the detection cone
                if abs(angle_diff) < half_width_angle:
                    goal_seen = True
                    confidence_bonus = detection.score
                    angle_bonus = 1 - (abs(angle_diff) / half_width_angle) # 1 at center, 0 at edge
                    distance = math.sqrt(x**2 + y**2)
                    distance_penalty = math.exp(-0.1 * distance) # Prefer closer areas

                    base_value += 500.0 * confidence_bonus * angle_bonus * distance_penalty

        # --- Fallback: Directional Guidance (if goal was not seen or no goal) ---
        if not goal_seen:
            # Convert world (x,y) to robot local frame around current robot position
            if self.current_pose is not None:
                dx = x - self.current_pose.position.x
                dy = y - self.current_pose.position.y
            else:
                dx, dy = x, y
            local_x = dx * math.cos(-current_yaw) - dy * math.sin(-current_yaw)
            local_y = dx * math.sin(-current_yaw) + dy * math.cos(-current_yaw)

            # Narrow-band directional preference: reward only a thin strip aligned with the desired axis
            gain = float(getattr(self, 'dir_longitudinal_gain', 10.0) or 10.0)
            sigma = float(getattr(self, 'dir_lateral_sigma_m', 0.2) or 0.2)  # lateral Gaussian sigma (m)

            # longitudinal = along desired direction; lateral = perpendicular
            if direction == 'forward':
                lon = max(0.0, local_x)
                lat = local_y
            elif direction == 'backward':
                lon = max(0.0, -local_x)
                lat = local_y
            elif direction == 'left':
                lon = max(0.0, local_y)
                lat = local_x
            elif direction == 'right':
                lon = max(0.0, -local_y)
                lat = local_x
            else:
                lon = max(0.0, local_x)
                lat = local_y

            # Gaussian weight across the lateral axis to form a narrow strip; cut off beyond ~3 sigma
            if sigma <= 1e-6:
                lateral_weight = 1.0 if abs(lat) < 1e-3 else 0.0
            else:
                lateral_weight = math.exp(-0.5 * (lat / sigma) * (lat / sigma))
                if abs(lat) > 3.0 * sigma:
                    lateral_weight = 0.0

            base_value += gain * lon * lateral_weight

            # Near-obstacle attractor: if the immediate cell ahead along the desired direction is an obstacle,
            # boost the current free cell to attract the robot to stop before the obstacle within the strip.
            if getattr(self, 'near_obstacle_boost_enable', True) and lateral_weight >= getattr(self, 'near_obstacle_lateral_min_weight', 0.1):
                try:
                    og = self.occupancy_grid
                    if og is not None:
                        step = float(getattr(self, 'near_obstacle_step_m', self.grid_resolution) or self.grid_resolution)
                        ox = og.info.origin.position.x
                        oy = og.info.origin.position.y
                        res = og.info.resolution
                        width = int(og.info.width)
                        height = int(og.info.height)

                        # Determine unit direction vector in world frame
                        cy = math.cos(current_yaw)
                        sy = math.sin(current_yaw)
                        if direction == 'forward':
                            ux, uy = cy, sy
                        elif direction == 'backward':
                            ux, uy = -cy, -sy
                        elif direction == 'left':
                            ux, uy = -sy, cy
                        elif direction == 'right':
                            ux, uy = sy, -cy
                        else:
                            ux, uy = cy, sy

                        ahead_x = x + step * ux
                        ahead_y = y + step * uy

                        ix_next = int(math.floor((ahead_x - ox) / res))
                        iy_next = int(math.floor((ahead_y - oy) / res))

                        if 0 <= ix_next < width and 0 <= iy_next < height:
                            idx_next = iy_next * width + ix_next
                            if og.data[idx_next] > 50:
                                boost = float(getattr(self, 'near_obstacle_boost_gain', 200.0) or 200.0)
                                base_value += boost * lateral_weight
                except Exception:
                    pass

        base_value += np.random.normal(0, 1.0)
        return base_value

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle



    def generate_path_points(self):
        """Generate path points based on value map"""
        if not self.value_map:
            return

        rospy.loginfo("Generating path points...")

        self.current_path_points = []
        values = np.array(self.value_map.data).reshape((self.value_map.info.height, self.value_map.info.width))

        # Parameters for path generation
        num_waypoints = 1
        suppression_radius_pixels = int(0.5 / self.value_map.info.resolution) # 0.5 meters

        temp_values = np.copy(values)

        for i in range(num_waypoints):
            # Find the coordinates of the max value in the map
            max_yx = np.unravel_index(np.argmax(temp_values), temp_values.shape)
            max_y, max_x = max_yx[0], max_yx[1]

            # Convert pixel coordinates to world coordinates
            world_x = self.value_map.info.origin.position.x + max_x * self.value_map.info.resolution
            world_y = self.value_map.info.origin.position.y + max_y * self.value_map.info.resolution

            # Create a new PathPoint
            point = PathPoint()
            point.header.stamp = rospy.Time.now()
            point.header.frame_id = "map" # Path points should be in the map frame
            point.position.x = world_x
            point.position.y = world_y
            point.position.z = 0.0
            point.orientation.w = 1.0  # Default orientation
            point.is_final_goal = (i == num_waypoints - 1)
            point.value = float(temp_values[max_y, max_x])
            point.position_tolerance = self.goal_tolerance
            point.orientation_tolerance = 0.2 # radians

            self.current_path_points.append(point)

            # Suppress the area around the chosen waypoint to pick diverse points
            min_y = max(0, max_y - suppression_radius_pixels)
            max_y_plus_1 = min(temp_values.shape[0], max_y + suppression_radius_pixels + 1)
            min_x = max(0, max_x - suppression_radius_pixels)
            max_x_plus_1 = min(temp_values.shape[1], max_x + suppression_radius_pixels + 1)
            temp_values[min_y:max_y_plus_1, min_x:max_x_plus_1] = -float('inf')

        # Sort path points based on distance from the robot's current position
        if self.current_pose:
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y
            self.current_path_points.sort(key=lambda p: (p.position.x - robot_x)**2 + (p.position.y - robot_y)**2)
        else:
            rospy.logwarn("No current pose available for path sorting, using origin.")
            self.current_path_points.sort(key=lambda p: p.position.x**2 + p.position.y**2)

        # Calculate orientation for each waypoint to face the next one
        if len(self.current_path_points) > 1:
            for i in range(len(self.current_path_points) - 1):
                p1 = self.current_path_points[i].position
                p2 = self.current_path_points[i+1].position
                yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
                quat = tfs.quaternion_from_euler(0, 0, yaw)
                self.current_path_points[i].orientation = Quaternion(*quat)
            # The last waypoint keeps its default orientation or inherits from the previous one
            if len(self.current_path_points) > 0:
                 self.current_path_points[-1].orientation = self.current_path_points[-2].orientation

        self.current_path_index = 0
        rospy.loginfo(f"Generated {len(self.current_path_points)} path points")

    def control_timer_callback(self, event):
        """Main control loop for navigation"""
        if not self.navigation_active or not self.current_path_points:
            return

        # Don't execute position control during scanning phase
        if self.scan_in_progress:
            return

        if self.current_path_index >= len(self.current_path_points):
            # Path completed
            self.navigation_completed()
            return

        current_waypoint = self.current_path_points[self.current_path_index]

        # Simple navigation to waypoint
        cmd = PositionCommand()
        cmd.position = current_waypoint.position
        cmd.velocity.x = 0.0
        cmd.velocity.y = 0.0
        cmd.velocity.z = 0.0
        # Always face the current target waypoint (robot pose -> waypoint)
        if self.current_pose is not None:
            dx = current_waypoint.position.x - self.current_pose.position.x
            dy = current_waypoint.position.y - self.current_pose.position.y
        # If in ArUco direct-navigation mode, continuously update body_goal until arrival
        if self.aruco_target_active and self.aruco_last_pose is not None:
            try:
                # Use the last seen pose in body frame to build a PositionCommand
                cmd = PositionCommand()
                cmd.position = self.aruco_last_pose.position
                cmd.velocity.x = 0.0
                cmd.velocity.y = 0.0
                cmd.velocity.z = 0.0
                try:
                    cmd.yaw = math.atan2(cmd.position.y, cmd.position.x)
                except Exception:
                    cmd.yaw = 0.0
                cmd.yaw_dot = 0.0
                self.controller_body_pub.publish(cmd)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Aruco direct-drive publish failed: {e}")

            # Check arrival
            if self._check_aruco_arrival():
                rospy.loginfo("Arrived near ArUco marker. Marking navigation complete.")
                self.aruco_target_active = False
                self.navigation_completed()
            return

            cmd.yaw = math.atan2(dy, dx)
        else:
            # Fallback to waypoint's stored orientation
            cmd.yaw = self.get_yaw_from_quaternion(current_waypoint.orientation)
        cmd.yaw_dot = 0.0

        self.controller_discrete_pub.publish(cmd)

        # Check if waypoint is reached
        if self.current_pose:
            dist_sq = (current_waypoint.position.x - self.current_pose.position.x)**2 + \
                      (current_waypoint.position.y - self.current_pose.position.y)**2
            if dist_sq < self.goal_tolerance**2:
                self.current_path_index += 1
                rospy.loginfo(f"Waypoint {self.current_path_index - 1} reached")
        else:
            rospy.logwarn_throttle(5.0, "No odometry available to check waypoint status.")

    def handle_aruco_detection(self):
        """Handle ArUco marker detection during navigation"""
        if not self.aruco_detections:
            return

        count = len(self.aruco_detections)
        rospy.loginfo(f"Handling ArUco detection during navigation... count={count}")

        # Case 1: exactly one marker -> go directly without querying VLM
        if count == 1:
            self.navigation_active = False
            try:
                self.controller_continuous_pub.publish(Twist())
            except Exception:
                pass
            self.navigate_to_aruco(self.aruco_detections[0])
            return

        # Case 2: multiple markers -> pause and ask VLM which one
        self.navigation_active = False
        try:
            self.controller_continuous_pub.publish(Twist())
        except Exception:
            pass

        if hasattr(self, 'latest_image'):
            query_data = {
                "type": "vision_query",
                "need_image": True,
                "image_available": True,
                "aruco_markers": [{"id": marker.id, "pose": {
                    "x": marker.pose.position.x,
                    "y": marker.pose.position.y,
                    "z": marker.pose.position.z
                }} for marker in self.aruco_detections],
                "query": "Which ArUco marker corresponds to the navigation target? Please answer JSON with keys 'target_found' (bool) and 'target_aruco_id' (int)."
            }

            query_msg = String()
            query_msg.data = json.dumps(query_data)
            self.vlm_query_pub.publish(query_msg)

    def navigation_completed(self):
        """Handle navigation completion"""
        rospy.loginfo("Navigation completed!")
        self.navigation_active = False

        # Actively stop the robot to avoid drift after completion
        try:
            self.controller_continuous_pub.publish(Twist())
        except Exception:
            pass

        # Notify state machine
        feedback_msg = String()
        feedback_msg.data = json.dumps({
            "status": "navigation_complete",
            "subtask_completed": True
        })
        self.status_feedback_pub.publish(feedback_msg)

        # If this subtask is the last one, publish final /status=0 after a short delay
        try:
            total = len(self.current_subtasks) if self.current_subtasks else 0
            # Only schedule final completion when we actually had subtasks and finished the last one
            if total > 0 and (self.current_subtask_index + 1) >= total:
                rospy.Timer(rospy.Duration(2.0), self.final_completion_callback, oneshot=True)
        except Exception:
            pass

    def handle_vlm_response(self, response_data):
        """Handle VLM response about ArUco markers with compatibility for 'vision_result' replies."""
        try:
            response = json.loads(response_data)
        except json.JSONDecodeError as e:
            rospy.logerr(f"Failed to parse VLM response: {e}")
            # Continue navigation on error
            self.navigation_active = True
            return

        # Primary expected schema
        if response.get("target_found", False):
            target_aruco_id = response.get("target_aruco_id")
            if target_aruco_id is not None:
                rospy.loginfo(f"Target found near ArUco marker {target_aruco_id}")
                for marker in self.aruco_detections:
                    if marker.id == target_aruco_id:
                        self.navigate_to_aruco(marker)
                        return
            # Missing ID: fall back to continue navigation
            rospy.logwarn("target_found=True but no target_aruco_id provided; continuing navigation")
            self.navigation_active = True
            return

        # Compatibility: try infer an ArUco ID from 'vision_result'
        inferred_id = None
        try:
            vr = response.get("vision_result", None)
            if isinstance(vr, list):
                text_chunks = []
                for item in vr:
                    if isinstance(item, str):
                        text_chunks.append(item)
                    elif isinstance(item, dict):
                        for k in ("id", "marker", "marker_id", "aruco_id"):
                            if k in item and isinstance(item[k], (int, str)):
                                text_chunks.append(str(item[k]))
                    else:
                        text_chunks.append(str(item))
                combined = " ".join(text_chunks)
                if combined:
                    try:
                        ids = re.findall(r"\b(\d{1,3})\b", combined)
                        ids_int = [int(x) for x in ids] if ids else []
                    except Exception:
                        ids_int = []
                    if ids_int:
                        current_ids = [m.id for m in self.aruco_detections] if self.aruco_detections else []
                        for cid in ids_int:
                            if cid in current_ids:
                                inferred_id = cid
                                break
                        if inferred_id is None and len(ids_int) == 1:
                            inferred_id = ids_int[0]
            # If still none and exactly one detection present, choose it
            if inferred_id is None and self.aruco_detections and len(self.aruco_detections) == 1:
                inferred_id = self.aruco_detections[0].id
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"vision_result compatibility parse failed: {e}")

        if inferred_id is not None:
            rospy.loginfo(f"Inferred target ArUco id {inferred_id} from vision_result; navigating to it")
            for marker in self.aruco_detections:
                if marker.id == inferred_id:
                    self.navigate_to_aruco(marker)
                    return

        # Default: continue navigation
        rospy.loginfo("No target found/inferred; continuing navigation")
        self.navigation_active = True

    def navigate_to_aruco(self, marker):
        """Navigate directly to an ArUco marker using body frame goal (base_link).
        Do NOT immediately publish final status; keep following until within arrival distance.
        """
        rospy.loginfo(f"Navigating to ArUco marker {marker.id} using body_goal (direct mode)")

        # Build initial body-frame PositionCommand
        cmd = PositionCommand()
        cmd.position = marker.pose.position
        cmd.velocity.x = 0.0
        cmd.velocity.y = 0.0
        cmd.velocity.z = 0.0
        try:
            cmd.yaw = math.atan2(marker.pose.position.y, marker.pose.position.x)
        except Exception:
            cmd.yaw = 0.0
        cmd.yaw_dot = 0.0
        # Publish once immediately
        self.controller_body_pub.publish(cmd)

        # Track target and enable direct-navigation mode
        self.aruco_target_active = True
        self.aruco_target_id = marker.id
        self.aruco_last_pose = marker.pose
        self.aruco_close_counter = 0
        # Ensure navigation loop runs
        self.navigation_active = True

    def _check_aruco_arrival(self):
        """Check whether we've reached the ArUco target (in body frame)."""
        if not self.aruco_target_active or not self.aruco_last_pose:
            return False
        try:
            dx = float(self.aruco_last_pose.position.x)
            dy = float(self.aruco_last_pose.position.y)
            dist = math.hypot(dx, dy)
        except Exception:
            return False
        # Require being within arrival distance for a few consecutive checks to avoid flicker
        if dist <= self.aruco_arrival_distance:
            self.aruco_close_counter += 1
        else:
            self.aruco_close_counter = 0
        return self.aruco_close_counter >= 3

        # yaw: relative rotation in body frame to face the target point
        try:
            cmd.yaw = math.atan2(marker.pose.position.y, marker.pose.position.x)
        except Exception:
            cmd.yaw = 0.0
        cmd.yaw_dot = 0.0

        # Publish to body_goal so controller converts to world frame
        self.controller_body_pub.publish(cmd)

        # Set flag that we found the final target
        self.task_completed = True

        # Wait a bit then notify completion
        rospy.Timer(rospy.Duration(5.0), self.final_completion_callback, oneshot=True)

    def final_completion_callback(self, event):
        """Handle final task completion"""
        rospy.loginfo("Final target reached! Task completed.")

        # Publish final status
        status_msg = Int32()
        status_msg.data = 0  # Task completed successfully
        self.final_status_pub.publish(status_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('core_node', anonymous=True)
        core_node = CoreNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass