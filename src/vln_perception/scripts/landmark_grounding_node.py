#!/usr/bin/env python3

import json

import rospy
from std_msgs.msg import String
from vln_core.config.models import Mission, MissionStep
from vln_msgs.msg import LandmarkObservation, Mission as MissionMsg
from vln_perception.plugins.loader import load_grounding_backend


class LandmarkGroundingNode:
    def __init__(self) -> None:
        self.backend_name = rospy.get_param("~backend", "mock")
        self.compiled_topic = rospy.get_param("~compiled_topic", "/vln/mission/compiled")
        self.output_topic = rospy.get_param("~output_topic", "/vln/perception/landmarks")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.debug_boxes_topic = rospy.get_param("~debug_boxes_topic", "/vln/perception/debug_boxes")
        self.backend = load_grounding_backend(self.backend_name)

        self.pub = rospy.Publisher(self.output_topic, LandmarkObservation, queue_size=20)
        self.debug_pub = rospy.Publisher(self.debug_boxes_topic, String, queue_size=5, latch=True)
        self.sub = rospy.Subscriber(self.compiled_topic, MissionMsg, self._on_mission, queue_size=10)
        self.reset_sub = rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=10)

    def _on_mission(self, msg: MissionMsg) -> None:
        mission = Mission(
            mission_id=msg.mission_id,
            raw_instruction=msg.raw_instruction,
            steps=[
                MissionStep(
                    step_index=step.step_index,
                    action=step.action,
                    target_label=step.target_label,
                    desired_yaw_rad=step.desired_yaw_rad,
                    desired_distance_m=step.desired_distance_m,
                    terminal=step.terminal,
                )
                for step in msg.steps
            ],
        )
        for item in self.backend.observe_mission(mission):
            observation = LandmarkObservation()
            observation.header.stamp = rospy.Time.now()
            observation.label = item.label
            observation.position.x = item.x
            observation.position.y = item.y
            observation.position.z = item.z
            observation.confidence = item.confidence
            observation.source = item.source
            observation.yaw_rad = item.yaw_rad
            self.pub.publish(observation)
        self._publish_debug_boxes(mission)

    def _publish_debug_boxes(self, mission: Mission) -> None:
        payload = {
            "mission_id": mission.mission_id,
            "boxes": list(self.backend.debug_boxes_for_mission(mission)),
        }
        self.debug_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _on_reset(self, _msg: String) -> None:
        self.debug_pub.publish(String(data=json.dumps({"mission_id": "", "boxes": []}, sort_keys=True)))


def main() -> None:
    rospy.init_node("landmark_grounding_node")
    LandmarkGroundingNode()
    rospy.spin()


if __name__ == "__main__":
    main()
