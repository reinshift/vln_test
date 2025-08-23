#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import tf.transformations as tfs
from threading import Lock

# ROS Messages
from geometry_msgs.msg import Twist, Point, Quaternion, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

# Custom Messages
from magv_vln_msgs.msg import PositionCommand, PathPoint

class Controller:
    def __init__(self):
        rospy.loginfo("Controller initializing...")

        # State variables
        self.current_pose = None
        self.current_velocity = None
        self.target_pose = None
        self.control_active = False

        # Thread safety
        self.control_lock = Lock()

        # Control parameters
        self.position_kp = rospy.get_param('~position_kp', 1.0)
        self.position_ki = rospy.get_param('~position_ki', 0.0)
        self.position_kd = rospy.get_param('~position_kd', 0.1)
        self.orientation_kp = rospy.get_param('~orientation_kp', 2.0)
        self.orientation_ki = rospy.get_param('~orientation_ki', 0.0)
        self.orientation_kd = rospy.get_param('~orientation_kd', 0.1)
        self.orientation_ki = rospy.get_param('~orientation_ki', 0.0)
        self.orientation_kd = rospy.get_param('~orientation_kd', 0.1)
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 1.0)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 1.0)
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.1)
        self.orientation_tolerance = rospy.get_param('~orientation_tolerance', 0.1)

        self.position_integral_clamp = rospy.get_param('~position_integral_clamp', 1.0)
        self.orientation_integral_clamp = rospy.get_param('~orientation_integral_clamp', 0.5)
        # PID error tracking
        self.position_error_integral = Point()
        self.position_error_prev = Point()
        self.orientation_error_integral = 0.0
        self.orientation_error_prev = 0.0

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/magv/omni_drive_controller/cmd_vel', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/controller_status', PoseStamped, queue_size=10)

        # Subscribers
        self.odometry_sub = rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback, queue_size=1)
        self.world_goal_sub = rospy.Subscriber('/world_goal', PositionCommand, self.world_goal_callback, queue_size=1)
        self.body_goal_sub = rospy.Subscriber('/body_goal', PositionCommand, self.body_goal_callback, queue_size=1)
        self.velocity_goal_sub = rospy.Subscriber('/velocity_goal', Twist, self.velocity_goal_callback, queue_size=1)
        self.path_point_sub = rospy.Subscriber('/path_point', PathPoint, self.path_point_callback, queue_size=1)

        # Control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20Hz control loop

        rospy.loginfo("Controller initialized")

    def odometry_callback(self, msg):
        """Handle odometry updates"""
        with self.control_lock:
            self.current_pose = msg.pose.pose
            self.current_velocity = msg.twist.twist

    def world_goal_callback(self, msg):
        """Handle world coordinate goal commands"""
        with self.control_lock:
            rospy.loginfo(f"Received world goal: x={msg.position.x:.2f}, y={msg.position.y:.2f}, yaw={msg.yaw:.2f}")

            # Convert to target pose
            target_pose = PoseStamped()
            target_pose.header.stamp = rospy.Time.now()
            target_pose.header.frame_id = "map"
            target_pose.pose.position = msg.position

            # Convert yaw to quaternion
            quat = tfs.quaternion_from_euler(0, 0, msg.yaw)
            target_pose.pose.orientation.x = quat[0]
            target_pose.pose.orientation.y = quat[1]
            target_pose.pose.orientation.z = quat[2]
            target_pose.pose.orientation.w = quat[3]

            self.target_pose = target_pose.pose
            self.control_active = True
            self.reset_pid_errors()

    def body_goal_callback(self, msg):
        """Handle body coordinate goal commands"""
        if not self.current_pose:
            rospy.logwarn("No current pose available for body coordinate transformation")
            return

        with self.control_lock:
            rospy.loginfo(f"Received body goal: x={msg.position.x:.2f}, y={msg.position.y:.2f}, yaw={msg.yaw:.2f}")

            # Transform body coordinates to world coordinates
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

            # Rotate body coordinates to world frame
            cos_yaw = math.cos(current_yaw)
            sin_yaw = math.sin(current_yaw)

            world_x = self.current_pose.position.x + (msg.position.x * cos_yaw - msg.position.y * sin_yaw)
            world_y = self.current_pose.position.y + (msg.position.x * sin_yaw + msg.position.y * cos_yaw)
            world_yaw = current_yaw + msg.yaw

            # Create target pose
            target_pose = PoseStamped()
            target_pose.header.stamp = rospy.Time.now()
            target_pose.header.frame_id = "map"
            target_pose.pose.position.x = world_x
            target_pose.pose.position.y = world_y
            target_pose.pose.position.z = self.current_pose.position.z + msg.position.z

            # Convert yaw to quaternion
            quat = tfs.quaternion_from_euler(0, 0, world_yaw)
            target_pose.pose.orientation.x = quat[0]
            target_pose.pose.orientation.y = quat[1]
            target_pose.pose.orientation.z = quat[2]
            target_pose.pose.orientation.w = quat[3]

            self.target_pose = target_pose.pose
            self.control_active = True
            self.reset_pid_errors()

    def velocity_goal_callback(self, msg):
        """Handle direct velocity commands"""
        rospy.loginfo(f"Received velocity goal: linear=({msg.linear.x:.2f}, {msg.linear.y:.2f}), angular={msg.angular.z:.2f}")

        # Directly publish velocity command
        self.cmd_vel_pub.publish(msg)

        # Disable position control when using velocity control
        with self.control_lock:
            self.control_active = False

    def path_point_callback(self, msg):
        """Handle path point commands"""
        with self.control_lock:
            rospy.loginfo(f"Received path point: x={msg.position.x:.2f}, y={msg.position.y:.2f}")

            # Convert PathPoint to target pose
            if msg.header.frame_id and msg.header.frame_id != "map":
                rospy.logwarn(f"PathPoint received with unexpected frame_id: '{msg.header.frame_id}'. Assuming 'map'.")

            target_pose = PoseStamped()
            target_pose.header.stamp = rospy.Time.now()
            target_pose.header.frame_id = "map" # Ensure frame is correct
            target_pose.pose.position = msg.position
            target_pose.pose.orientation = msg.orientation

            self.target_pose = target_pose.pose
            self.control_active = True
            self.reset_pid_errors()

    def control_loop(self, event):
        """Main control loop"""
        if not self.control_active or not self.current_pose or not self.target_pose:
            return

        with self.control_lock:
            # Calculate position error
            pos_error = Point()
            pos_error.x = self.target_pose.position.x - self.current_pose.position.x
            pos_error.y = self.target_pose.position.y - self.current_pose.position.y
            pos_error.z = 0.0  # Ignore z for ground robot

            # Calculate orientation error
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
            target_yaw = self.get_yaw_from_quaternion(self.target_pose.orientation)
            orientation_error = self.normalize_angle(target_yaw - current_yaw)

            # Check if goal is reached
            position_distance = math.sqrt(pos_error.x**2 + pos_error.y**2)
            if (position_distance < self.position_tolerance and
                abs(orientation_error) < self.orientation_tolerance):

                # Goal reached, stop the robot
                self.control_active = False
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.loginfo("Goal reached!")
                return

            # PID control for position
            try:
                dt = (event.current_real - event.last_real).to_sec()
                if dt <= 0:
                    return # Avoid division by zero or negative dt
            except AttributeError:
                # Fallback for initial call where last_real might not be available
                dt = 0.05

            # Position PID
            self.position_error_integral.x += pos_error.x * dt
            self.position_error_integral.y += pos_error.y * dt

            # Clamp integral term to prevent windup
            self.position_error_integral.x = max(-self.position_integral_clamp, min(self.position_integral_clamp, self.position_error_integral.x))
            self.position_error_integral.y = max(-self.position_integral_clamp, min(self.position_integral_clamp, self.position_error_integral.y))

            pos_error_derivative = Point()
            pos_error_derivative.x = (pos_error.x - self.position_error_prev.x) / dt
            pos_error_derivative.y = (pos_error.y - self.position_error_prev.y) / dt

            # Calculate control commands
            cmd_vel = Twist()

            # Linear velocity (in world frame)
            world_vel_x = (self.position_kp * pos_error.x +
                          self.position_ki * self.position_error_integral.x +
                          self.position_kd * pos_error_derivative.x)
            world_vel_y = (self.position_kp * pos_error.y +
                          self.position_ki * self.position_error_integral.y +
                          self.position_kd * pos_error_derivative.y)

            # Transform to body frame
            cos_yaw = math.cos(current_yaw)
            sin_yaw = math.sin(current_yaw)

            cmd_vel.linear.x = world_vel_x * cos_yaw + world_vel_y * sin_yaw
            cmd_vel.linear.y = -world_vel_x * sin_yaw + world_vel_y * cos_yaw

            # Orientation PID
            self.orientation_error_integral += orientation_error * dt
            self.orientation_error_integral = max(-self.orientation_integral_clamp, min(self.orientation_integral_clamp, self.orientation_error_integral))
            orientation_error_derivative = (orientation_error - self.orientation_error_prev) / dt

            cmd_vel.angular.z = (self.orientation_kp * orientation_error +
                               self.orientation_ki * self.orientation_error_integral +
                               self.orientation_kd * orientation_error_derivative)

            # Apply velocity limits
            cmd_vel.linear.x = max(-self.max_linear_vel, min(self.max_linear_vel, cmd_vel.linear.x))
            cmd_vel.linear.y = max(-self.max_linear_vel, min(self.max_linear_vel, cmd_vel.linear.y))
            cmd_vel.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, cmd_vel.angular.z))

            # Publish command
            self.cmd_vel_pub.publish(cmd_vel)

            # Update previous errors
            self.position_error_prev = pos_error
            self.orientation_error_prev = orientation_error

            # Publish status
            status_msg = PoseStamped()
            status_msg.header.stamp = rospy.Time.now()
            status_msg.header.frame_id = "map"
            status_msg.pose = self.target_pose
            self.status_pub.publish(status_msg)

    def get_yaw_from_quaternion(self, quaternion):
        """Extract yaw angle from quaternion"""
        euler = tfs.euler_from_quaternion([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        ])
        return euler[2]  # yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset_pid_errors(self):
        """Reset PID error accumulators"""
        self.position_error_integral = Point()
        self.position_error_prev = Point()
        self.orientation_error_integral = 0.0
        self.orientation_error_prev = 0.0

if __name__ == '__main__':
    try:
        rospy.init_node('controller', anonymous=True)
        controller = Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
