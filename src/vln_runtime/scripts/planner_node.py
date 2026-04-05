#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from vln_core.planning.scoring import build_trajectory
from vln_core.config.models import Pose2D, SemanticLandmark
from vln_runtime.conversions import mission_from_msg, runtime_state_msg, trajectory_to_msg, yaw_from_quaternion
from vln_msgs.msg import Mission as MissionMsg, RuntimeState, SemanticLandmark as SemanticLandmarkMsg, Trajectory


class PlannerNode:
    def __init__(self) -> None:
        self.compiled_topic = rospy.get_param("~compiled_topic", "/vln/mission/compiled")
        self.semantic_map_topic = rospy.get_param("~semantic_map_topic", "/vln/world/semantic_map")
        self.trajectory_topic = rospy.get_param("~trajectory_topic", "/vln/planning/trajectory")
        self.runtime_state_topic = rospy.get_param("~runtime_state_topic", "/vln/runtime/state")
        self.odom_topic = rospy.get_param("~odom_topic", "/sim/uav/odom")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.feedback_topic = rospy.get_param("~feedback_topic", "/vln/runtime/controller_feedback")
        self.plan_rate_hz = float(rospy.get_param("~plan_rate_hz", 2.0))
        self.republish_sec = float(rospy.get_param("~republish_sec", 3.0))

        self.landmarks = {}
        self.active_mission = None
        self.current_pose = Pose2D(x=0.0, y=0.0, z=1.5, yaw_rad=0.0)
        self.last_signature = ""
        self.last_publish_sec = 0.0
        self.landmark_revision = 0
        self.mission_revision = 0

        self.trajectory_pub = rospy.Publisher(self.trajectory_topic, Trajectory, queue_size=10, latch=True)
        self.state_pub = rospy.Publisher(self.runtime_state_topic, RuntimeState, queue_size=20)
        self.mission_sub = rospy.Subscriber(self.compiled_topic, MissionMsg, self._on_mission, queue_size=10)
        self.landmark_sub = rospy.Subscriber(self.semantic_map_topic, SemanticLandmarkMsg, self._on_landmark, queue_size=20)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=20)
        self.reset_sub = rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=20)
        self.feedback_sub = rospy.Subscriber(self.feedback_topic, RuntimeState, self._on_feedback, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.plan_rate_hz)), self._tick)

    def _on_landmark(self, msg: SemanticLandmarkMsg) -> None:
        updated = SemanticLandmark(
            label=msg.label,
            x=msg.position.x,
            y=msg.position.y,
            z=msg.position.z,
            confidence=msg.confidence,
            observation_count=msg.observation_count,
            state=msg.state,
        )
        previous = self.landmarks.get(msg.label)
        self.landmarks[msg.label] = updated
        if previous != updated:
            self.landmark_revision += 1

    def _on_mission(self, msg: MissionMsg) -> None:
        self.active_mission = mission_from_msg(msg)
        self.mission_revision += 1
        self.last_signature = ""

    def _on_odom(self, msg: Odometry) -> None:
        self.current_pose = Pose2D(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            z=msg.pose.pose.position.z or 1.5,
            yaw_rad=yaw_from_quaternion(msg.pose.pose.orientation),
        )

    def _on_reset(self, _msg: String) -> None:
        self.active_mission = None
        self.last_signature = ""
        self.last_publish_sec = 0.0
        self.state_pub.publish(runtime_state_msg("", 0, "planner_reset", "planner state cleared", status_code=0))

    def _on_feedback(self, msg: RuntimeState) -> None:
        if self.active_mission is None or msg.mission_id != self.active_mission.mission_id:
            return
        if msg.mission_complete or msg.phase == "mission_complete":
            self.active_mission = None
            self.last_signature = ""
            self.state_pub.publish(
                runtime_state_msg(msg.mission_id, msg.active_step_index, "planner_idle", "mission completed")
            )

    def _signature(self, trajectory) -> str:
        rounded_waypoints = [
            (round(point.x, 3), round(point.y, 3), round(point.z, 3), round(point.yaw_rad, 3))
            for point in trajectory.waypoints
        ]
        return repr(
            (
                trajectory.mission_id,
                self.mission_revision,
                self.landmark_revision,
                rounded_waypoints,
            )
        )

    def _tick(self, _event) -> None:
        if self.active_mission is None:
            return

        trajectory = build_trajectory(
            self.active_mission,
            self.landmarks.values(),
            start_pose=self.current_pose,
        )
        signature = self._signature(trajectory)
        now_sec = rospy.get_time()
        should_publish = signature != self.last_signature or (now_sec - self.last_publish_sec) >= self.republish_sec
        if not should_publish:
            return

        self.trajectory_pub.publish(trajectory_to_msg(trajectory))
        self.state_pub.publish(
            runtime_state_msg(
                self.active_mission.mission_id,
                1,
                "trajectory_planned",
                f"waypoints={len(trajectory.waypoints)} landmarks={len(self.landmarks)}",
            )
        )
        self.last_signature = signature
        self.last_publish_sec = now_sec


def main() -> None:
    rospy.init_node("planner_node")
    PlannerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
