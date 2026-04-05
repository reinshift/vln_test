#!/usr/bin/env python3

import json
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from vln_msgs.msg import RuntimeState

try:
    from mavros_msgs.msg import State as MavrosState
    from mavros_msgs.srv import CommandBool, SetMode
except ImportError:  # pragma: no cover - exercised only on machines without MAVROS
    MavrosState = None
    CommandBool = None
    SetMode = None


def _yaw_to_quaternion(yaw_rad: float):
    half = float(yaw_rad) / 2.0
    return math.sin(half), math.cos(half)


class Px4UavBridgeNode:
    def __init__(self) -> None:
        self.command_topic = rospy.get_param("~command_topic", "/sim/uav/cmd_vel")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.runtime_state_topic = rospy.get_param("~runtime_state_topic", "/vln/runtime/state")
        self.sim_odom_topic = rospy.get_param("~odom_topic", "/sim/uav/odom")
        self.rgb_topic = rospy.get_param("~rgb_topic", "/sim/uav/rgb/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/sim/uav/depth/image_raw")
        self.scan_topic = rospy.get_param("~scan_topic", "/sim/uav/scan")

        self.mavros_state_topic = rospy.get_param("~mavros_state_topic", "/mavros/state")
        self.mavros_odom_source_topic = rospy.get_param("~mavros_odom_source_topic", "/mavros/local_position/odom")
        self.velocity_setpoint_topic = rospy.get_param(
            "~velocity_setpoint_topic",
            "/mavros/setpoint_velocity/cmd_vel_unstamped",
        )
        self.set_mode_service = rospy.get_param("~set_mode_service", "/mavros/set_mode")
        self.arm_service = rospy.get_param("~arm_service", "/mavros/cmd/arming")

        self.rgb_source_topic = rospy.get_param("~rgb_source_topic", "/camera/rgb/image_raw")
        self.depth_source_topic = rospy.get_param("~depth_source_topic", "")
        self.scan_source_topic = rospy.get_param("~scan_source_topic", "/scan")

        self.stream_rate_hz = float(rospy.get_param("~stream_rate_hz", 20.0))
        self.climb_speed_mps = float(rospy.get_param("~climb_speed_mps", 0.8))
        self.altitude_tolerance_m = float(rospy.get_param("~altitude_tolerance_m", 0.15))
        self.offboard_warmup_cycles = int(rospy.get_param("~offboard_warmup_cycles", 50))

        self.last_cmd = Twist()
        self.latest_odom = None
        self.target_altitude = float(rospy.get_param("~default_altitude", 1.5))
        self.connected = False
        self.armed = False
        self.mode = ""
        self.mission_complete = False
        self._warmup_cycles = 0
        self._last_mode_request_sec = 0.0
        self._last_arm_request_sec = 0.0

        self.setpoint_pub = rospy.Publisher(self.velocity_setpoint_topic, Twist, queue_size=20)
        self.odom_pub = rospy.Publisher(self.sim_odom_topic, Odometry, queue_size=20)
        self.rgb_pub = rospy.Publisher(self.rgb_topic, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=10)
        self.scan_pub = rospy.Publisher(self.scan_topic, LaserScan, queue_size=10)

        rospy.Subscriber(self.command_topic, Twist, self._on_command, queue_size=20)
        rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=20)
        rospy.Subscriber(self.runtime_state_topic, RuntimeState, self._on_runtime_state, queue_size=20)
        rospy.Subscriber(self.mavros_odom_source_topic, Odometry, self._on_mavros_odom, queue_size=20)
        rospy.Subscriber(self.rgb_source_topic, Image, self._on_rgb, queue_size=10)
        if self.depth_source_topic:
            rospy.Subscriber(self.depth_source_topic, Image, self._on_depth, queue_size=10)
        rospy.Subscriber(self.scan_source_topic, LaserScan, self._on_scan, queue_size=10)

        if MavrosState is not None:
            rospy.Subscriber(self.mavros_state_topic, MavrosState, self._on_mavros_state, queue_size=20)
            self.arm_proxy = rospy.ServiceProxy(self.arm_service, CommandBool)
            self.mode_proxy = rospy.ServiceProxy(self.set_mode_service, SetMode)
        else:
            self.arm_proxy = None
            self.mode_proxy = None
            rospy.logwarn("mavros_msgs is unavailable; PX4 bridge will publish setpoints but cannot arm or switch modes.")

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.stream_rate_hz)), self._tick)

    def _on_command(self, msg: Twist) -> None:
        self.last_cmd = msg

    def _on_reset(self, msg: String) -> None:
        payload = {}
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            payload = {}
        spawn = payload.get("spawn", {})
        self.target_altitude = float(spawn.get("z", self.target_altitude))
        self.last_cmd = Twist()
        self.mission_complete = False
        self._warmup_cycles = 0

    def _on_runtime_state(self, msg: RuntimeState) -> None:
        if msg.mission_complete or msg.phase in {"mission_complete", "mission_failed"}:
            self.mission_complete = True

    def _on_mavros_state(self, msg) -> None:
        self.connected = bool(msg.connected)
        self.armed = bool(msg.armed)
        self.mode = str(msg.mode)

    def _on_mavros_odom(self, msg: Odometry) -> None:
        self.latest_odom = msg
        outgoing = Odometry()
        outgoing.header = msg.header
        outgoing.child_frame_id = msg.child_frame_id or "base_link"
        outgoing.pose = msg.pose
        outgoing.twist = msg.twist
        if not outgoing.header.frame_id:
            outgoing.header.frame_id = "map"
        self.odom_pub.publish(outgoing)

    def _on_rgb(self, msg: Image) -> None:
        self.rgb_pub.publish(msg)

    def _on_depth(self, msg: Image) -> None:
        self.depth_pub.publish(msg)

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan_pub.publish(msg)

    def _current_altitude(self) -> float:
        if self.latest_odom is None:
            return 0.0
        return float(self.latest_odom.pose.pose.position.z)

    def _desired_command(self) -> Twist:
        outgoing = Twist()
        if self.mission_complete:
            return outgoing

        if self.latest_odom is None or self._current_altitude() < (self.target_altitude - self.altitude_tolerance_m):
            outgoing.linear.z = self.climb_speed_mps
            return outgoing

        outgoing.linear.x = self.last_cmd.linear.x
        outgoing.linear.y = self.last_cmd.linear.y
        outgoing.angular.z = self.last_cmd.angular.z
        return outgoing

    def _call_mode(self, now_sec: float) -> None:
        if self.mode_proxy is None or (now_sec - self._last_mode_request_sec) < 2.0:
            return
        self._last_mode_request_sec = now_sec
        try:
            self.mode_proxy(base_mode=0, custom_mode="OFFBOARD")
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(5.0, "OFFBOARD mode request failed: %s", exc)

    def _call_arm(self, now_sec: float) -> None:
        if self.arm_proxy is None or (now_sec - self._last_arm_request_sec) < 2.0:
            return
        self._last_arm_request_sec = now_sec
        try:
            self.arm_proxy(True)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(5.0, "PX4 arm request failed: %s", exc)

    def _tick(self, _event) -> None:
        setpoint = self._desired_command()
        self.setpoint_pub.publish(setpoint)

        if self.latest_odom is None:
            synthetic = Odometry()
            synthetic.header.stamp = rospy.Time.now()
            synthetic.header.frame_id = "map"
            synthetic.child_frame_id = "base_link"
            synthetic.pose.pose.position.z = self.target_altitude
            synthetic.pose.pose.orientation.z, synthetic.pose.pose.orientation.w = _yaw_to_quaternion(0.0)
            self.odom_pub.publish(synthetic)

        if not self.connected:
            return

        self._warmup_cycles += 1
        if self._warmup_cycles < max(1, self.offboard_warmup_cycles):
            return

        now_sec = rospy.get_time()
        if self.mode != "OFFBOARD":
            self._call_mode(now_sec)
        if not self.armed:
            self._call_arm(now_sec)


def main() -> None:
    rospy.init_node("px4_uav_bridge_node")
    Px4UavBridgeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
