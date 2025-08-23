#!/usr/bin/env python3
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
from magv_vln_msgs.msg import VehicleStatus

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
        self.emergency_stop_distance = rospy.get_param('~emergency_stop_distance', 0.5) # meters

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
        
        # Current position
        self.current_position = Point()
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

    def pointcloud_callback(self, msg):
        """Handle pointcloud updates for emergency stop detection"""
        self.navigation_ready = True  # Assume navigation is ready if we get pointclouds

        if self.current_velocity.linear.x <= 0.1:  # Only check for obstacles when moving forward
            if self.emergency_stop_active:
                rospy.loginfo("Emergency Stop Released: Robot is not moving forward.")
                self.emergency_stop_active = False
                self.current_state = VehicleStatus.STATE_NAVIGATION
            return

        obstacle_detected = False
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            if 0 < point[0] < self.emergency_stop_distance and abs(point[1]) < 0.4:  # Check a narrow corridor
                obstacle_detected = True
                break

        if obstacle_detected:
            if not self.emergency_stop_active:
                rospy.logwarn("Emergency Stop Triggered: Obstacle detected!")
                self.emergency_stop_active = True
                self.current_state = VehicleStatus.STATE_EMERGENCY_STOP
                self.emergency_stop_pub.publish(Twist())  # Send zero velocity
        else:
            if self.emergency_stop_active:
                rospy.loginfo("Emergency Stop Released: Path is clear.")
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
            status_msg.diagnostic_info = self.get_diagnostic_info()
            
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
