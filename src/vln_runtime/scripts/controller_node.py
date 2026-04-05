#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from vln_core.config.models import Pose2D, TrajectoryPlan, Waypoint
from vln_core.planning.follower import TrajectoryFollower
from vln_runtime.conversions import runtime_state_msg, yaw_from_quaternion
from vln_msgs.msg import RuntimeState, Trajectory


class ControllerNode:
    def __init__(self) -> None:
        self.trajectory_topic = rospy.get_param("~trajectory_topic", "/vln/planning/trajectory")
        self.command_topic = rospy.get_param("~command_topic", "/vln/control/command")
        self.runtime_state_topic = rospy.get_param("~runtime_state_topic", "/vln/runtime/state")
        self.odom_topic = rospy.get_param("~odom_topic", "/sim/uav/odom")
        self.feedback_topic = rospy.get_param("~feedback_topic", "/vln/runtime/controller_feedback")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.control_rate_hz = float(rospy.get_param("~control_rate_hz", 10.0))
        self.max_speed = float(rospy.get_param("~max_speed", 1.0))
        self.waypoint_tolerance_m = float(rospy.get_param("~waypoint_tolerance_m", 0.6))

        self.current_pose = Pose2D(x=0.0, y=0.0, yaw_rad=0.0, z=1.5)
        self.current_mission_id = ""
        self.plan_signature = ""
        self.last_state_publish_sec = 0.0
        self.follower = TrajectoryFollower(waypoint_tolerance_m=self.waypoint_tolerance_m)
        self.command_pub = rospy.Publisher(self.command_topic, Twist, queue_size=10)
        self.state_pub = rospy.Publisher(self.runtime_state_topic, RuntimeState, queue_size=20)
        self.feedback_pub = rospy.Publisher(self.feedback_topic, RuntimeState, queue_size=20)
        self.trajectory_sub = rospy.Subscriber(self.trajectory_topic, Trajectory, self._on_trajectory, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=20)
        self.reset_sub = rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.control_rate_hz)), self._tick)

    def _on_odom(self, msg: Odometry) -> None:
        self.current_pose = Pose2D(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            z=msg.pose.pose.position.z or 1.5,
            yaw_rad=yaw_from_quaternion(msg.pose.pose.orientation),
        )
        self._emit_progress_events()

    def _on_trajectory(self, msg: Trajectory) -> None:
        plan = TrajectoryPlan(
            mission_id=msg.mission_id,
            frame_id=msg.frame_id or "map",
            waypoints=[
                Waypoint(
                    x=pose.pose.position.x,
                    y=pose.pose.position.y,
                    z=pose.pose.position.z,
                    yaw_rad=yaw_from_quaternion(pose.pose.orientation),
                )
                for pose in msg.waypoints
            ],
            target_speed=msg.target_speed or 1.0,
            planner_mode=msg.planner_mode,
        )
        signature = self._signature(plan)
        if signature == self.plan_signature:
            return
        self.follower.set_plan(plan)
        self.current_mission_id = plan.mission_id
        self.plan_signature = signature
        self.state_pub.publish(
            runtime_state_msg(
                plan.mission_id,
                self.follower.active_waypoint_index + 1 if plan.waypoints else 0,
                "control_tracking",
                f"loaded trajectory with {len(plan.waypoints)} waypoints",
            )
        )
        self._emit_progress_events()

    def _on_reset(self, msg: String) -> None:
        self.current_mission_id = ""
        self.plan_signature = ""
        self.follower.reset()
        self.command_pub.publish(Twist())
        self.state_pub.publish(runtime_state_msg("", 0, "controller_reset", msg.data or "reset_event", status_code=0))

    def _signature(self, plan: TrajectoryPlan) -> str:
        return repr(
            (
                plan.mission_id,
                [(round(point.x, 3), round(point.y, 3), round(point.z, 3), round(point.yaw_rad, 3)) for point in plan.waypoints],
                round(plan.target_speed, 3),
                plan.planner_mode,
            )
        )

    def _emit_progress_events(self) -> None:
        events = self.follower.update_pose(self.current_pose)
        for event in events:
            if event["type"] == "waypoint_reached":
                state = runtime_state_msg(
                    event["mission_id"],
                    int(event["waypoint_index"]),
                    "waypoint_reached",
                    f"waypoint {event['waypoint_index']} reached",
                )
            else:
                self.command_pub.publish(Twist())
                state = runtime_state_msg(
                    event["mission_id"],
                    len(self.follower.plan.waypoints) if self.follower.plan is not None else 0,
                    "mission_complete",
                    "controller reached final waypoint",
                    mission_complete=True,
                )
            self.feedback_pub.publish(state)
            self.state_pub.publish(state)

    def _tick(self, _event) -> None:
        if self.follower.plan is None or self.follower.mission_complete:
            return

        shaped = self.follower.compute_command(self.current_pose, max_speed=self.max_speed)
        cmd = Twist()
        cmd.linear.x = shaped["linear_x"]
        cmd.linear.y = shaped["linear_y"]
        cmd.angular.z = shaped["angular_z"]
        self.command_pub.publish(cmd)

        now_sec = rospy.get_time()
        if (now_sec - self.last_state_publish_sec) < 1.0:
            return
        self.state_pub.publish(
            runtime_state_msg(
                self.current_mission_id,
                self.follower.active_waypoint_index + 1 if self.follower.plan and self.follower.plan.waypoints else 0,
                "control_tracking",
                "tracking active waypoint",
            )
        )
        self.last_state_publish_sec = now_sec


def main() -> None:
    rospy.init_node("controller_node")
    ControllerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
