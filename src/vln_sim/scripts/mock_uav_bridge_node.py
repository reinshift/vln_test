#!/usr/bin/env python3

import json
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String


class MockUavBridgeNode:
    def __init__(self) -> None:
        self.command_topic = rospy.get_param("~command_topic", "/sim/uav/cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "/sim/uav/odom")
        self.scan_topic = rospy.get_param("~scan_topic", "/sim/uav/scan")
        self.rgb_topic = rospy.get_param("~rgb_topic", "/sim/uav/rgb/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/sim/uav/depth/image_raw")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.loop_hz = float(rospy.get_param("~loop_hz", 10.0))
        self.default_altitude = float(rospy.get_param("~default_altitude", 1.5))
        self.clear_range_m = float(rospy.get_param("~clear_range_m", 12.0))

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = self.default_altitude
        self.pose_yaw = 0.0
        self.last_cmd = Twist()

        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)
        self.scan_pub = rospy.Publisher(self.scan_topic, LaserScan, queue_size=10)
        self.rgb_pub = rospy.Publisher(self.rgb_topic, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=10)

        self.command_sub = rospy.Subscriber(self.command_topic, Twist, self._on_command, queue_size=20)
        self.reset_sub = rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.loop_hz)), self._tick)

    def _on_command(self, msg: Twist) -> None:
        self.last_cmd = msg

    def _on_reset(self, _msg: String) -> None:
        payload = {}
        try:
            payload = json.loads(_msg.data) if _msg.data else {}
        except json.JSONDecodeError:
            payload = {}

        spawn = payload.get("spawn", {})
        self.pose_x = float(spawn.get("x", 0.0))
        self.pose_y = float(spawn.get("y", 0.0))
        self.pose_z = float(spawn.get("z", self.default_altitude))
        self.pose_yaw = float(spawn.get("yaw", 0.0))
        self.last_cmd = Twist()

    def _tick(self, _event) -> None:
        dt = 1.0 / max(1.0, self.loop_hz)
        self.pose_yaw += self.last_cmd.angular.z * dt
        self.pose_x += math.cos(self.pose_yaw) * self.last_cmd.linear.x * dt
        self.pose_y += math.sin(self.pose_yaw) * self.last_cmd.linear.x * dt

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.pose_x
        odom.pose.pose.position.y = self.pose_y
        odom.pose.pose.position.z = self.pose_z
        odom.pose.pose.orientation.z = math.sin(self.pose_yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.pose_yaw / 2.0)
        odom.twist.twist = self.last_cmd
        self.odom_pub.publish(odom)

        scan = LaserScan()
        scan.header.stamp = odom.header.stamp
        scan.header.frame_id = "base_link"
        scan.angle_min = -1.57
        scan.angle_max = 1.57
        scan.angle_increment = 0.157
        scan.range_min = 0.2
        scan.range_max = self.clear_range_m
        scan.ranges = [self.clear_range_m] * 21
        self.scan_pub.publish(scan)

        image = Image()
        image.header.stamp = odom.header.stamp
        image.height = 1
        image.width = 1
        image.encoding = "mono8"
        image.step = 1
        image.data = b"\x00"
        self.rgb_pub.publish(image)
        self.depth_pub.publish(image)


def main() -> None:
    rospy.init_node("mock_uav_bridge_node")
    MockUavBridgeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
