#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from vln_core.config.models import SemanticObservation
from vln_core.world_model.semantic_map import SemanticMapStore
from vln_runtime.conversions import runtime_state_msg, semantic_landmark_to_msg
from vln_msgs.msg import LandmarkObservation, RuntimeState, SemanticLandmark


class SemanticMapNode:
    def __init__(self) -> None:
        self.observation_topic = rospy.get_param("~observation_topic", "/vln/perception/landmarks")
        self.semantic_map_topic = rospy.get_param("~semantic_map_topic", "/vln/world/semantic_map")
        self.runtime_state_topic = rospy.get_param("~runtime_state_topic", "/vln/runtime/state")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.snapshot_rate_hz = float(rospy.get_param("~snapshot_rate_hz", 1.0))

        self.store = SemanticMapStore()
        self.map_pub = rospy.Publisher(self.semantic_map_topic, SemanticLandmark, queue_size=20)
        self.state_pub = rospy.Publisher(self.runtime_state_topic, RuntimeState, queue_size=20)
        self.sub = rospy.Subscriber(self.observation_topic, LandmarkObservation, self._on_observation, queue_size=20)
        self.reset_sub = rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(0.2, self.snapshot_rate_hz)), self._publish_snapshot)

    def _on_observation(self, msg: LandmarkObservation) -> None:
        item = self.store.observe(
            SemanticObservation(
                label=msg.label,
                x=msg.position.x,
                y=msg.position.y,
                z=msg.position.z,
                confidence=msg.confidence,
                source=msg.source,
                yaw_rad=msg.yaw_rad,
            )
        )
        self.map_pub.publish(semantic_landmark_to_msg(item))
        self.state_pub.publish(runtime_state_msg("", 0, "semantic_map_update", f"tracked={len(self.store.snapshot())}"))

    def _on_reset(self, msg: String) -> None:
        self.store.reset()
        self.state_pub.publish(runtime_state_msg("", 0, "semantic_map_reset", msg.data or "reset_event", status_code=0))

    def _publish_snapshot(self, _event) -> None:
        snapshot = self.store.snapshot()
        if not snapshot:
            return
        for item in snapshot:
            self.map_pub.publish(semantic_landmark_to_msg(item))


def main() -> None:
    rospy.init_node("semantic_map_node")
    SemanticMapNode()
    rospy.spin()


if __name__ == "__main__":
    main()
