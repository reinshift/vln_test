#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from vln_core.safety.corridor import summarize_scan
from vln_runtime.conversions import runtime_state_msg
from vln_msgs.msg import RuntimeState, SafetyEvent


class SafetySupervisorNode:
    def __init__(self) -> None:
        self.scan_topic = rospy.get_param("~scan_topic", "/sim/uav/scan")
        self.command_topic = rospy.get_param("~command_topic", "/vln/control/command")
        self.safe_command_topic = rospy.get_param("~safe_command_topic", "/sim/uav/cmd_vel")
        self.event_topic = rospy.get_param("~event_topic", "/vln/safety/event")
        self.runtime_state_topic = rospy.get_param("~runtime_state_topic", "/vln/runtime/state")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.stop_distance_m = float(rospy.get_param("~stop_distance_m", 1.0))
        self.block_streak_required = int(rospy.get_param("~block_streak_required", 2))
        self.clear_streak_required = int(rospy.get_param("~clear_streak_required", 2))

        self.latest_summary = summarize_scan([], 0.0, 0.0)
        self.blocked_active = False
        self.blocked_streak = 0
        self.clear_streak = 0
        self.safe_pub = rospy.Publisher(self.safe_command_topic, Twist, queue_size=10)
        self.event_pub = rospy.Publisher(self.event_topic, SafetyEvent, queue_size=20)
        self.state_pub = rospy.Publisher(self.runtime_state_topic, RuntimeState, queue_size=20)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self._on_scan, queue_size=20)
        self.command_sub = rospy.Subscriber(self.command_topic, Twist, self._on_command, queue_size=20)
        self.reset_sub = rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=20)

    def _on_scan(self, msg: LaserScan) -> None:
        self.latest_summary = summarize_scan(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            stop_distance_m=self.stop_distance_m,
        )
        previous = self.blocked_active
        if self.latest_summary.blocked:
            self.blocked_streak += 1
            self.clear_streak = 0
            if self.blocked_streak >= max(1, self.block_streak_required):
                self.blocked_active = True
        else:
            self.blocked_streak = 0
            self.clear_streak += 1
            if self.clear_streak >= max(1, self.clear_streak_required):
                self.blocked_active = False

        event = SafetyEvent()
        event.header.stamp = rospy.Time.now()
        event.level = "stop" if self.blocked_active else "info"
        event.reason = self.latest_summary.reason
        event.clearance_m = self.latest_summary.clearance_m
        event.stop_commanded = self.blocked_active
        self.event_pub.publish(event)
        if previous != self.blocked_active:
            phase = "safety_stop" if self.blocked_active else "safety_clear"
            detail = self.latest_summary.reason if self.blocked_active else "corridor cleared"
            self.state_pub.publish(runtime_state_msg("", 0, phase, detail, status_code=2 if self.blocked_active else 1))

    def _on_command(self, msg: Twist) -> None:
        outgoing = Twist()
        outgoing.linear.x = msg.linear.x
        outgoing.linear.y = msg.linear.y
        outgoing.angular.z = msg.angular.z
        phase = "safety_pass"
        detail = "command forwarded"
        if self.blocked_active:
            outgoing = Twist()
            phase = "safety_stop"
            detail = self.latest_summary.reason
        self.safe_pub.publish(outgoing)
        self.state_pub.publish(runtime_state_msg("", 0, phase, detail, status_code=2 if self.blocked_active else 1))

    def _on_reset(self, msg: String) -> None:
        self.latest_summary = summarize_scan([], 0.0, 0.0)
        self.blocked_active = False
        self.blocked_streak = 0
        self.clear_streak = 0
        self.safe_pub.publish(Twist())
        self.state_pub.publish(runtime_state_msg("", 0, "safety_reset", msg.data or "reset_event", status_code=0))


def main() -> None:
    rospy.init_node("safety_supervisor_node")
    SafetySupervisorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
