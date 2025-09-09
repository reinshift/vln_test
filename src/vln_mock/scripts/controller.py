#!/catkin_ws/venv310/bin/python3
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
from magv_vln_msgs.msg import PositionCommand, PathPoint, VehicleStatus

class Controller:
    def __init__(self):
        rospy.loginfo("Controller initializing...")

        # State variables
        self.current_pose = None
        self.current_velocity = None
        self.target_pose = None
        self.pending_target_pose = None  # cache goals arriving before NAVIGATION
        self.control_active = False
        self.vln_state = None  # gate control by VLN state

        # Thread safety
        self.control_lock = Lock()

        # Control parameters
        self.position_kp = rospy.get_param('~position_kp', 1.0)
        self.position_ki = rospy.get_param('~position_ki', 0.0)
        self.position_kd = rospy.get_param('~position_kd', 0.1)
        self.orientation_kp = rospy.get_param('~orientation_kp', 2.0)
        self.orientation_ki = rospy.get_param('~orientation_ki', 0.0)
        self.orientation_kd = rospy.get_param('~orientation_kd', 0.1)

        # Alignment strategy: rotate to target yaw before translating (helps camera align with path)
        self.align_before_translate = rospy.get_param('~align_before_translate', True)
        self.align_yaw_threshold = rospy.get_param('~align_yaw_threshold', 0.3)  # radians, ~17 deg
        # Optional: scale linear speed by heading alignment (0..1). Disabled by default
        self.scale_linear_by_heading = rospy.get_param('~scale_linear_by_heading', False)
        # Velocity and acceleration limits
        # Defaults set to: v <= 1.5 m/s, w <= 6.28 rad/s; a <= 3 m/s^2, alpha <= 6.28 rad/s^2
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 1.5)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 6.28)
        self.max_linear_accel = rospy.get_param('~max_linear_accel', 3.0)
        self.max_angular_accel = rospy.get_param('~max_angular_accel', 6.28)
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.1)
        self.orientation_tolerance = rospy.get_param('~orientation_tolerance', 0.3)

        # Safety option: publish zero cmd_vel periodically during INITIALIZING state only
        self.enable_init_zero = rospy.get_param('~enable_init_zero', True)
        self.init_zero_rate = rospy.get_param('~init_zero_rate', 5.0)  # Hz
        self.init_zero_angular_guard = rospy.get_param('~init_zero_angular_guard', 0.05)  # rad/s, skip zeroing if actively rotating

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
        self.vln_status_sub = rospy.Subscriber('/vln_status', VehicleStatus, self.vln_status_callback, queue_size=1)

        # Control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20Hz control loop

        # Zero-velocity safety timer (active only in INITIALIZING when enabled)
        try:
            period = 1.0 / max(0.1, float(self.init_zero_rate))
        except Exception:
            period = 0.2
        self.init_zero_timer = rospy.Timer(rospy.Duration(max(0.05, period)), self.init_zero_loop)

        rospy.loginfo("Controller initialized")

        # Track last published command for acceleration limiting
        from geometry_msgs.msg import Twist as _Twist
        self.last_cmd_vel = _Twist()

    def odometry_callback(self, msg):
        """Handle odometry updates"""
        with self.control_lock:
            self.current_pose = msg.pose.pose
            self.current_velocity = msg.twist.twist

    def vln_status_callback(self, msg):
        """Track VLN state for gating controls"""
        with self.control_lock:
            old_state = self.vln_state
            state_changed = (old_state != msg.state)
            if state_changed:
                rospy.loginfo("VLN state changed: %s -> %s (%s -> %d)",
                              self.state_name(old_state), self.state_name(msg.state),
                              str(old_state) if old_state is not None else "None", msg.state)
            # If leaving NAVIGATION, actively brake and disarm controller
            if old_state == VehicleStatus.STATE_NAVIGATION and msg.state != VehicleStatus.STATE_NAVIGATION:
                zero = Twist()
                self.cmd_vel_pub.publish(zero)
                self.last_cmd_vel = zero
                self.control_active = False
                rospy.loginfo("Exited NAVIGATION: published zero cmd_vel and disarmed controller")

            self.vln_state = msg.state
            # If we have a target already but were waiting for NAVIGATION, arm control now
            if self.is_navigation() and self.target_pose is not None and not self.control_active:
                rospy.loginfo("Entering NAVIGATION with a pending goal; arming controller now")
                self.control_active = True
                self.reset_pid_errors()

    def is_navigation(self):
        return self.vln_state == VehicleStatus.STATE_NAVIGATION

    def world_goal_callback(self, msg):
        """Handle world coordinate goal commands"""
        # Build a Pose from incoming command
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = "map"
        target_pose.pose.position = msg.position
        quat = tfs.quaternion_from_euler(0, 0, msg.yaw)
        target_pose.pose.orientation.x = quat[0]
        target_pose.pose.orientation.y = quat[1]
        target_pose.pose.orientation.z = quat[2]
        target_pose.pose.orientation.w = quat[3]

        with self.control_lock:
            if not self.is_navigation():
                # Cache pending goal to execute when NAVIGATION starts
                self.pending_target_pose = target_pose.pose
                rospy.logwarn("Received world goal while not in NAVIGATION; cached as pending goal")
                return

            rospy.loginfo(f"Received world goal: x={msg.position.x:.2f}, y={msg.position.y:.2f}, yaw={msg.yaw:.2f}")
            self.target_pose = target_pose.pose
            self.control_active = True
            self.reset_pid_errors()
            # Clear any pending since we are acting on a fresh goal
            self.pending_target_pose = None

    def body_goal_callback(self, msg):
        """Handle body coordinate goal commands"""
        if not self.current_pose:
            rospy.logwarn("No current pose available for body coordinate transformation")
            return

        # First compute target in world frame
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        world_x = self.current_pose.position.x + (msg.position.x * cos_yaw - msg.position.y * sin_yaw)
        world_y = self.current_pose.position.y + (msg.position.x * sin_yaw + msg.position.y * cos_yaw)
        world_yaw = current_yaw + msg.yaw

        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = "map"
        target_pose.pose.position.x = world_x
        target_pose.pose.position.y = world_y
        target_pose.pose.position.z = self.current_pose.position.z + msg.position.z
        quat = tfs.quaternion_from_euler(0, 0, world_yaw)
        target_pose.pose.orientation.x = quat[0]
        target_pose.pose.orientation.y = quat[1]
        target_pose.pose.orientation.z = quat[2]
        target_pose.pose.orientation.w = quat[3]

        with self.control_lock:
            if not self.is_navigation():
                # Cache pending body goal (converted to world pose) until NAVIGATION
                self.pending_target_pose = target_pose.pose
                rospy.logwarn("Received body goal while not in NAVIGATION; cached as pending goal")
                return

            rospy.loginfo(f"Received body goal: x={msg.position.x:.2f}, y={msg.position.y:.2f}, yaw={msg.yaw:.2f}")
            self.target_pose = target_pose.pose
            self.control_active = True
            self.reset_pid_errors()
            self.pending_target_pose = None

    def velocity_goal_callback(self, msg):
        """Handle direct velocity commands"""
        rospy.loginfo(f"Received velocity goal: linear=({msg.linear.x:.2f}, {msg.linear.y:.2f}), angular={msg.angular.z:.2f}")

        # Gate: during non-NAVIGATION, allow rotation for scan but zero linear components
        if not self.is_navigation():
            sanitized = Twist()
            sanitized.linear.x = 0.0
            sanitized.linear.y = 0.0
            # clamp angular to configured limit
            omega = max(-self.max_angular_vel, min(self.max_angular_vel, msg.angular.z))
            sanitized.angular.z = omega
            self.cmd_vel_pub.publish(sanitized)
            rospy.loginfo_throttle(1.0, "Sanitized velocity in non-NAVIGATION: zeroed linear, kept angular")
            with self.control_lock:
                self.control_active = False
                self.last_cmd_vel = sanitized
            return

        # In NAVIGATION, pass through
        self.cmd_vel_pub.publish(msg)
        # Track last command for guards/diagnostics
        self.last_cmd_vel = msg

        # Disable position control when using velocity control
        with self.control_lock:
            self.control_active = False

    def path_point_callback(self, msg):
        """Handle path point commands"""
        with self.control_lock:
            if not self.is_navigation():
                rospy.logwarn("Ignoring path point since VLN state != NAVIGATION")
                return
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
        # If we just entered NAVIGATION and have a pending goal, arm it
        if self.is_navigation() and not self.control_active and self.pending_target_pose is not None:
            with self.control_lock:
                self.target_pose = self.pending_target_pose
                self.pending_target_pose = None
                self.control_active = True
                self.reset_pid_errors()
                rospy.loginfo("Armed pending goal at control loop start")

        if not self.control_active or not self.current_pose or not self.target_pose or not self.is_navigation():
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

            # If required, align yaw before translating: when heading error is large, stop linear motion
            if self.align_before_translate and abs(orientation_error) > self.align_yaw_threshold:
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
            elif self.scale_linear_by_heading:
                # Optionally scale linear speed by cos of heading error to reduce sideways drift
                scale = max(0.0, math.cos(abs(orientation_error)))
                cmd_vel.linear.x *= scale
                cmd_vel.linear.y *= scale

            # Orientation PID
            self.orientation_error_integral += orientation_error * dt
            self.orientation_error_integral = max(-self.orientation_integral_clamp, min(self.orientation_integral_clamp, self.orientation_error_integral))
            orientation_error_derivative = (orientation_error - self.orientation_error_prev) / dt

            cmd_vel.angular.z = (self.orientation_kp * orientation_error +
                               self.orientation_ki * self.orientation_error_integral +
                               self.orientation_kd * orientation_error_derivative)

            # Apply velocity limits (caps)
            cmd_vel.linear.x = max(-self.max_linear_vel, min(self.max_linear_vel, cmd_vel.linear.x))
            cmd_vel.linear.y = max(-self.max_linear_vel, min(self.max_linear_vel, cmd_vel.linear.y))
            cmd_vel.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, cmd_vel.angular.z))

            # Apply acceleration limits (rate limiting per control step)
            # Compute dt from timer event for accurate rate limiting
            try:
                dt = (event.current_real - event.last_real).to_sec()
                if dt <= 0:
                    dt = 0.05
            except Exception:
                dt = 0.05

            max_dv = self.max_linear_accel * dt
            max_dw = self.max_angular_accel * dt

            # Limit change relative to last published command
            def clamp_delta(current, last, max_delta):
                delta = current - last
                if delta > max_delta:
                    return last + max_delta
                if delta < -max_delta:
                    return last - max_delta
                return current

            cmd_vel.linear.x = clamp_delta(cmd_vel.linear.x, self.last_cmd_vel.linear.x, max_dv)
            cmd_vel.linear.y = clamp_delta(cmd_vel.linear.y, self.last_cmd_vel.linear.y, max_dv)
            cmd_vel.angular.z = clamp_delta(cmd_vel.angular.z, self.last_cmd_vel.angular.z, max_dw)

            # Publish command
            self.cmd_vel_pub.publish(cmd_vel)

            # Throttled diagnostics for yaw control
            rospy.loginfo_throttle(1.0, "yaw_err=%.3f, ang_z=%.3f, lin=(%.3f, %.3f)" % (
                orientation_error, cmd_vel.angular.z, cmd_vel.linear.x, cmd_vel.linear.y))

            # Save for next iteration's acceleration limiting
            self.last_cmd_vel = cmd_vel

            # Update previous errors
            self.position_error_prev = pos_error
            self.orientation_error_prev = orientation_error

            # Publish status
            status_msg = PoseStamped()
            status_msg.header.stamp = rospy.Time.now()
            status_msg.header.frame_id = "map"
            status_msg.pose = self.target_pose
            self.status_pub.publish(status_msg)

    def init_zero_loop(self, event):
        """Publish zero velocity periodically in INITIALIZING state for safety (optional)."""
        if not self.enable_init_zero:
            return
        # In any non-NAVIGATION state, publish zero at a low rate as a safety brake
        if self.vln_state != VehicleStatus.STATE_NAVIGATION:
            # Do not override active rotation during INITIALIZING scanning
            if self.vln_state == VehicleStatus.STATE_INITIALIZING and abs(self.last_cmd_vel.angular.z) > self.init_zero_angular_guard:
                return
            zero = Twist()
            self.cmd_vel_pub.publish(zero)
            rospy.loginfo_throttle(2.0, "Publishing zero cmd_vel in non-NAVIGATION state")

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

    def state_name(self, state_code):
        """Map VehicleStatus code to readable name"""
        if state_code == VehicleStatus.STATE_INITIALIZING:
            return "INITIALIZING"
        if state_code == VehicleStatus.STATE_EXPLORATION:
            return "EXPLORATION"
        if state_code == VehicleStatus.STATE_NAVIGATION:
            return "NAVIGATION"
        if state_code == VehicleStatus.STATE_IDLE:
            return "IDLE"
        if state_code == VehicleStatus.STATE_ERROR:
            return "ERROR"
        if state_code == VehicleStatus.STATE_EMERGENCY_STOP:
            return "EMERGENCY_STOP"
        return "UNKNOWN"

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
