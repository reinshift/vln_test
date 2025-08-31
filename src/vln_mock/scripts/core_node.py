#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import rospy
import json
import numpy as np
import cv2
import math
from threading import Lock
from collections import deque
import tf.transformations as tfs

# ROS Messages
from std_msgs.msg import String, Int32, Bool
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, Twist, PoseStamped, Quaternion
from sensor_msgs.msg import CompressedImage

# Custom Messages
from magv_vln_msgs.msg import (
    VehicleStatus, ValueMap, PathPoint, PositionCommand,
    DetectedObjectArray, Detection2DArray
)
from aruco_detector.msg import ArucoInfo, ArucoMarker

class CoreNode:
    def __init__(self):
        rospy.loginfo("Core Node initializing...")

        # State variables
        self.current_state = VehicleStatus.STATE_IDLE
        self.current_subtasks = []
        self.current_subtask_index = 0
        self.current_subtask = None
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
        self.current_yaw = 0.0
        self.current_pose = None

        # Thread safety
        self.state_lock = Lock()

        # Publishers
        self.value_map_pub = rospy.Publisher('/value_map', ValueMap, queue_size=1)
        self.path_point_pub = rospy.Publisher('/path_point', PathPoint, queue_size=10)
        self.controller_discrete_pub = rospy.Publisher('/world_goal', PositionCommand, queue_size=10)
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

        # Camera parameters for projecting detections
        self.image_width = rospy.get_param('~image_width', 1080)
        self.horizontal_fov = rospy.get_param('~horizontal_fov', 2.0) # radians
        self.f_x = (self.image_width / 2.0) / np.tan(self.horizontal_fov / 2.0)

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
                    self.current_subtasks = json.loads(msg.current_subtask_json)
                    self.current_subtask_index = msg.current_subtask_index
                    if 0 <= self.current_subtask_index < len(self.current_subtasks):
                        self.current_subtask = self.current_subtasks[self.current_subtask_index]
                        # Reset scan and plan state for new subtask
                        self.scan_and_plan_complete = False
                        self.directional_detections = {}  # Clear previous detections
                    rospy.loginfo(f"Received subtasks: {len(self.current_subtasks)} tasks, current index: {self.current_subtask_index}")
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

        # 2. Command the robot to perform a 360-degree scan
        rospy.loginfo("Commanding 360-degree rotation for scanning...")
        scan_cmd = Twist()
        # Rotate at max allowable angular speed; controller will enforce acceleration limits
        scan_cmd.angular.z = self.max_angular_vel
        # Send via velocity_goal so Controller applies limits
        try:
            self.controller_velocity_pub.publish(scan_cmd)
        except Exception:
            # Fallback to direct cmd_vel if velocity_goal unavailable
            self.controller_continuous_pub.publish(scan_cmd)

        # 3. Use a timer to stop the scan and trigger planning
        # Fixed scan duration (configurable); rotate at max angular speed
        rospy.Timer(rospy.Duration(self.scan_duration_sec), self.finish_scan_and_plan, oneshot=True)

    def finish_scan_and_plan(self, event):
        """
        Called by a timer after the robot has finished its 360-degree scan.
        Stops the robot, computes the map, generates a path, and notifies the state machine.
        """
        rospy.loginfo("Rotation finished. Stopping robot and starting planning.")

        # 1. Stop the robot's rotation
        stop_cmd = Twist()
        try:
            self.controller_velocity_pub.publish(stop_cmd)
        except Exception:
            self.controller_continuous_pub.publish(stop_cmd)

        # Clear the GroundingDINO prompt
        self.dino_prompt_pub.publish(String(data=""))

        if self.occupancy_grid is None:
            rospy.logerr("Cannot plan without an occupancy grid!")
            # Optionally, you could add logic here to notify the state machine of a failure.
            return

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

    def dino_detections_callback(self, msg):
        """
        Store detections from GroundingDINO, associating them with the current yaw.
        This is crucial for the rotational scan.
        """
        # Only store detections if we are in the initialization (scanning) phase
        if self.current_state != VehicleStatus.STATE_INITIALIZING:
            return

        for detection in msg.detections:
            label = detection.label
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
                direction = value
                break
        goal = self.current_subtask.get('goal', None)

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

        value_map.data = values.tolist()
        self.value_map = value_map
        self.value_map_pub.publish(value_map)

        rospy.loginfo("Value map computed and published")

    def compute_cell_value(self, x, y, direction, goal, current_yaw):
        """
        Compute value for a single cell. This now prioritizes directional detections
        from the rotational scan, creating high-value sectors in the direction
        where the goal object was seen.
        """
        base_value = 0.0
        goal_seen = False

        # --- Directional Value from Rotational Scan ---
        if goal and goal in self.directional_detections:
            cell_angle = math.atan2(y, x)

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
            # Convert cell position to robot's local frame for directional calculation
            local_x = x * math.cos(-current_yaw) - y * math.sin(-current_yaw)
            local_y = x * math.sin(-current_yaw) + y * math.cos(-current_yaw)

            if direction == 'forward':
                base_value += local_x * 10.0
            elif direction == 'backward':
                base_value -= local_x * 10.0
            elif direction == 'left':
                base_value += local_y * 10.0
            elif direction == 'right':
                base_value -= local_y * 10.0

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
        num_waypoints = 5
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

        rospy.loginfo("Handling ArUco detection during navigation...")

        # Stop the vehicle
        self.navigation_active = False
        stop_cmd = Twist()
        self.controller_continuous_pub.publish(stop_cmd)

        # Query VLM about detected ArUco markers
        if hasattr(self, 'latest_image'):
            query_data = {
                "image_available": True,
                "aruco_markers": [{"id": marker.id, "pose": {
                    "x": marker.pose.position.x,
                    "y": marker.pose.position.y,
                    "z": marker.pose.position.z
                }} for marker in self.aruco_detections],
                "query": "Are there any target objects near these ArUco markers?"
            }

            query_msg = String()
            query_msg.data = json.dumps(query_data)
            self.vlm_query_pub.publish(query_msg)

    def navigation_completed(self):
        """Handle navigation completion"""
        rospy.loginfo("Navigation completed!")
        self.navigation_active = False

        # Notify state machine
        feedback_msg = String()
        feedback_msg.data = json.dumps({
            "status": "navigation_complete",
            "subtask_completed": True
        })
        self.status_feedback_pub.publish(feedback_msg)

    def handle_vlm_response(self, response_data):
        """Handle VLM response about ArUco markers"""
        try:
            response = json.loads(response_data)
            if response.get("target_found", False):
                # Target found near ArUco marker
                target_aruco_id = response.get("target_aruco_id")
                rospy.loginfo(f"Target found near ArUco marker {target_aruco_id}")

                # Navigate directly to the target ArUco marker
                for marker in self.aruco_detections:
                    if marker.id == target_aruco_id:
                        self.navigate_to_aruco(marker)
                        break
            else:
                # No target found, continue navigation
                rospy.loginfo("No target found near ArUco markers, continuing navigation")
                self.navigation_active = True

        except json.JSONDecodeError as e:
            rospy.logerr(f"Failed to parse VLM response: {e}")
            # Continue navigation on error
            self.navigation_active = True

    def navigate_to_aruco(self, marker):
        """Navigate directly to an ArUco marker"""
        rospy.loginfo(f"Navigating to ArUco marker {marker.id}")

        # Create position command to go to ArUco marker
        cmd = PositionCommand()
        cmd.position = marker.pose.position
        cmd.velocity.x = 0.0
        cmd.velocity.y = 0.0
        cmd.velocity.z = 0.0
        # Calculate yaw to face the marker from the current position
        if self.current_pose:
            dx = marker.pose.position.x - self.current_pose.position.x
            dy = marker.pose.position.y - self.current_pose.position.y
            cmd.yaw = math.atan2(dy, dx)
        else:
            cmd.yaw = 0.0 # Fallback
        cmd.yaw_dot = 0.0

        self.controller_discrete_pub.publish(cmd)

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
