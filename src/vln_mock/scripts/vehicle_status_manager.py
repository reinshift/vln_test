#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import rospy
import json
from threading import Lock

# ROS Messages
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, Twist
import sensor_msgs.point_cloud2 as pc2

# Custom Messages
from magv_vln_msgs.msg import VehicleStatus, PositionCommand

class VehicleStatusManager:
    def __init__(self):
        rospy.init_node('vehicle_status_manager', anonymous=True)
        rospy.loginfo("Vehicle Status Manager initializing...")

        # State variables
        self.current_state = VehicleStatus.STATE_IDLE
        self.subtasks = []
        self.current_subtask_index = 0
        self.task_completed = False
        self.sensors_ready = False
        self.navigation_ready = False
        self.emergency_stop_active = False
        self.current_velocity = Twist()
        self.emergency_stop_distance = rospy.get_param('~emergency_stop_distance', 0.15) # meters
        self.emergency_stop_half_width = rospy.get_param('~emergency_stop_half_width', 0.25) # meters, corridor half width in y
        # Height filtering to suppress ground/overhead points
        self.min_obstacle_z = rospy.get_param('~min_obstacle_z', 0.15)  # meters, ignore ground points (raised from 0.05)
        self.max_obstacle_z = rospy.get_param('~max_obstacle_z', 1.5)   # meters, ignore high overhead points
        # Near-range and stability filtering for ES
        self.es_min_range = rospy.get_param('~es_min_range', 0.12)  # meters, ignore very close speckles
        self.es_min_points = rospy.get_param('~es_min_points', 3)   # require at least N points in corridor
        self.es_trigger_frames = rospy.get_param('~es_trigger_frames', 2)  # consecutive frames to trigger
        self.es_release_frames = rospy.get_param('~es_release_frames', 2)  # consecutive clear frames to release

        # Diagnostics: nearest distances from latest pointcloud
        self.min_distance_all = float('inf')
        self.min_distance_forward = float('inf')
        # ES stability counters
        self._es_hit_streak = 0
        self._es_clear_streak = 0

        self.forward_points_count = 0
        self.last_pc_frame = ''

        # Thread safety
        self.state_lock = Lock()

        # Publishers
        self.status_pub = rospy.Publisher('/vln_status', VehicleStatus, queue_size=10)
        self.emergency_stop_pub = rospy.Publisher('/magv/omni_drive_controller/cmd_vel', Twist, queue_size=1)

        # Subscribers
        self.subtasks_sub = rospy.Subscriber('/subtasks', String, self.subtasks_callback, queue_size=1)
        self.core_feedback_sub = rospy.Subscriber('/core_feedback', String, self.core_feedback_callback, queue_size=1)
        self.odometry_sub = rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback, queue_size=1)
        self.pointcloud_sub = rospy.Subscriber('/magv/scan/3d', PointCloud2, self.pointcloud_callback, queue_size=1)
        self.cmd_vel_sub = rospy.Subscriber('/magv/omni_drive_controller/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        # Subscribe to latest goal to reflect in status.target_position
        self.world_goal_sub = rospy.Subscriber('/world_goal', PositionCommand, self.world_goal_callback, queue_size=1)

        # Current position / target
        self.current_position = Point()
        self.latest_target_position = Point()
        self.odometry_frame_id = "map"  # Default frame_id

        # Status publishing timer
        self.status_timer = rospy.Timer(rospy.Duration(1.0), self.publish_status)

        rospy.loginfo("Vehicle Status Manager initialized")

    def subtasks_callback(self, msg):
        """Handle subtasks from VLM processor"""
        with self.state_lock:
            try:
                self.subtasks = json.loads(msg.data)
                self.current_subtask_index = 0
                self.task_completed = False

                rospy.loginfo(f"Received {len(self.subtasks)} subtasks")
                rospy.loginfo(f"Subtasks: {self.subtasks}")

                # Transition to initialization state
                self.current_state = VehicleStatus.STATE_INITIALIZING
                rospy.loginfo("State changed to INITIALIZING")

            except json.JSONDecodeError as e:
                rospy.logerr(f"Failed to parse subtasks: {e}")

    def core_feedback_callback(self, msg):
        """Handle feedback from core node"""
        with self.state_lock:
            try:
                feedback = json.loads(msg.data)
                status = feedback.get("status", "")

                if status == "initialization_complete":
                    # Core node finished initialization, switch to navigation
                    if self.current_state == VehicleStatus.STATE_INITIALIZING:
                        self.current_state = VehicleStatus.STATE_NAVIGATION
                        rospy.loginfo("State changed to NAVIGATION")

                elif status == "navigation_complete":
                    # Core node finished current subtask
                    subtask_completed = feedback.get("subtask_completed", False)

                    if subtask_completed:
                        self.current_subtask_index += 1

                        if self.current_subtask_index >= len(self.subtasks):
                            # All subtasks completed
                            self.task_completed = True
                            self.current_state = VehicleStatus.STATE_IDLE
                            rospy.loginfo("All subtasks completed!")
                        else:
                            # Move to next subtask
                            self.current_state = VehicleStatus.STATE_INITIALIZING
                            rospy.loginfo(f"Moving to subtask {self.current_subtask_index + 1}/{len(self.subtasks)}")

            except json.JSONDecodeError as e:
                rospy.logerr(f"Failed to parse core feedback: {e}")

    def odometry_callback(self, msg):
        """Handle odometry updates"""
        self.current_position = msg.pose.pose.position
        self.odometry_frame_id = msg.header.frame_id
        # Update sensors ready status
        self.sensors_ready = True

    def cmd_vel_callback(self, msg):
        """Store the latest velocity command"""
        if not self.emergency_stop_active:
            self.current_velocity = msg

    def world_goal_callback(self, msg):
        """Track the latest world goal for status.target_position"""
        # Directly store the position component
        self.latest_target_position = msg.position

    def pointcloud_callback(self, msg):
        """Handle pointcloud updates for emergency stop detection and diagnostics"""
        self.navigation_ready = True  # Assume navigation is ready if we get pointclouds

        # Reset diagnostics per frame
        self.min_distance_all = float('inf')
        self.min_distance_forward = float('inf')
        self.forward_points_count = 0
        self.last_pc_frame = msg.header.frame_id if hasattr(msg, 'header') else ''

        # Only check for obstacles if the robot has some translational velocity
        if abs(self.current_velocity.linear.x) < 0.1 and abs(self.current_velocity.linear.y) < 0.1:
            if self.emergency_stop_active:
                rospy.loginfo("Emergency Stop Released: Robot is not moving forward.")
                self.emergency_stop_active = False
                self.current_state = VehicleStatus.STATE_NAVIGATION
            # Still compute nearest distances for diagnostics
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = point[0], point[1], point[2]
                d = (x*x + y*y) ** 0.5
                if d < self.min_distance_all:
                    self.min_distance_all = d
                if (x > 0 and abs(y) < self.emergency_stop_half_width and self.min_obstacle_z <= z <= self.max_obstacle_z):
                    self.forward_points_count += 1
                    if x < self.min_distance_forward:
                        self.min_distance_forward = x
            return

        # When moving: compute distances and detect obstacles
        obstacle_detected = False
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point[0], point[1], point[2]
            d = (x*x + y*y) ** 0.5
            if d < self.min_distance_all:
                self.min_distance_all = d

            # Near-range speckle removal
            if d < self.es_min_range:
                continue

            # Forward corridor with height filtering
            in_forward_corridor = (
                0 < x < self.emergency_stop_distance and
                abs(y) < self.emergency_stop_half_width and
                self.min_obstacle_z <= z <= self.max_obstacle_z
            )
            if in_forward_corridor:
                self.forward_points_count += 1
                if x < self.min_distance_forward:
                    self.min_distance_forward = x
                # do not mark detected yet; use count+streak below

        # Apply point-count threshold to decide detection this frame
        if self.forward_points_count >= int(self.es_min_points):
            obstacle_detected = True

        # Streak logic for trigger/release stability
        if obstacle_detected:
            self._es_hit_streak += 1
            self._es_clear_streak = 0
        else:
            self._es_clear_streak += 1
            self._es_hit_streak = 0

        should_trigger = (self._es_hit_streak >= int(self.es_trigger_frames))
        should_release = (self._es_clear_streak >= int(self.es_release_frames))

        if should_trigger:
            if not self.emergency_stop_active:
                rospy.logwarn("Emergency Stop Triggered: Obstacle detected! (min_fwd=%.2f m, count=%d, hit_streak=%d)",
                              self.min_distance_forward if self.min_distance_forward != float('inf') else -1.0,
                              int(self.forward_points_count), self._es_hit_streak)
                self.emergency_stop_active = True
                self.current_state = VehicleStatus.STATE_EMERGENCY_STOP
                self.emergency_stop_pub.publish(Twist())  # Send zero velocity
        elif should_release:
            if self.emergency_stop_active:
                rospy.loginfo("Emergency Stop Released: Path is clear. (clear_streak=%d)", self._es_clear_streak)
                self.emergency_stop_active = False
                self.current_state = VehicleStatus.STATE_NAVIGATION

    def publish_status(self, event):
        """Publish current vehicle status"""
        with self.state_lock:
            status_msg = VehicleStatus()
            status_msg.header.stamp = rospy.Time.now()
            status_msg.header.frame_id = self.odometry_frame_id

            # Basic status
            status_msg.state = self.current_state
            status_msg.state_description = self.get_state_description()
            status_msg.is_moving = (self.current_state == VehicleStatus.STATE_NAVIGATION)
            status_msg.has_goal = len(self.subtasks) > 0 and not self.task_completed
            status_msg.sensors_ready = self.sensors_ready
            status_msg.navigation_ready = self.navigation_ready

            # Position
            status_msg.current_position = self.current_position
            # Reflect latest target if any
            status_msg.target_position = self.latest_target_position

            # Subtask information
            if self.subtasks:
                status_msg.current_subtask_json = json.dumps(self.subtasks)
                status_msg.current_subtask_index = self.current_subtask_index
                status_msg.total_subtasks = len(self.subtasks)

            # Task completion
            status_msg.task_completed = self.task_completed
            if self.task_completed:
                status_msg.completion_message = "All VLN subtasks completed successfully"

            # Diagnostic info
            # Combine diagnostics with nearest point info
            extra = self.get_diagnostic_info()
            nearest_all = ("inf" if self.min_distance_all == float('inf') else f"{self.min_distance_all:.2f}")
            nearest_fwd = ("inf" if self.min_distance_forward == float('inf') else f"{self.min_distance_forward:.2f}")
            status_msg.diagnostic_info = (
                f"{extra} | pc_frame={self.last_pc_frame} min_all={nearest_all}m min_forward_x={nearest_fwd}m "
                f"fwd_count={self.forward_points_count} es_dist={self.emergency_stop_distance:.2f} es_half={self.emergency_stop_half_width:.2f} "
                f"min_range={self.es_min_range:.2f} min_pts={self.es_min_points} hit_stk={self._es_hit_streak} clr_stk={self._es_clear_streak}"
            )

            self.status_pub.publish(status_msg)

    def get_state_description(self):
        """Get human-readable state description"""
        state_descriptions = {
            VehicleStatus.STATE_IDLE: "Idle - Waiting for instructions",
            VehicleStatus.STATE_INITIALIZING: "Initializing - Processing subtask",
            VehicleStatus.STATE_NAVIGATION: "Navigation - Executing movement",
            VehicleStatus.STATE_EXPLORATION: "Exploration - Searching environment",
            VehicleStatus.STATE_ERROR: "Error - System malfunction",
            VehicleStatus.STATE_EMERGENCY_STOP: "Emergency Stop - Safety halt"
        }
        return state_descriptions.get(self.current_state, "Unknown State")

    def get_diagnostic_info(self):
        """Get diagnostic information"""
        if not self.subtasks:
            return "No active tasks"

        if self.task_completed:
            return "All tasks completed successfully"

        current_subtask = None
        if 0 <= self.current_subtask_index < len(self.subtasks):
            current_subtask = self.subtasks[self.current_subtask_index]

        if current_subtask:
            direction = 'unknown'
            for key, value in current_subtask.items():
                if 'subtask' in key:
                    direction = value
                    break
            goal = current_subtask.get('goal', 'none')
            return f"Subtask {self.current_subtask_index + 1}/{len(self.subtasks)}: {direction} -> {goal}"

        return "Processing subtasks"

if __name__ == '__main__':
    try:
        status_manager = VehicleStatusManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass