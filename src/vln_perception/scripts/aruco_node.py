#!/usr/bin/env python3

import rospy
from vln_msgs.msg import LandmarkObservation


class ArucoNode:
    def __init__(self) -> None:
        self.enabled = bool(rospy.get_param("~enabled", False))
        self.output_topic = rospy.get_param("~output_topic", "/vln/perception/landmarks")
        self.marker_label = rospy.get_param("~marker_label", "aruco_marker")
        self.pub = rospy.Publisher(self.output_topic, LandmarkObservation, queue_size=10)
        if self.enabled:
            self.timer = rospy.Timer(rospy.Duration(5.0), self._tick)
        else:
            self.timer = None

    def _tick(self, _event) -> None:
        observation = LandmarkObservation()
        observation.header.stamp = rospy.Time.now()
        observation.label = self.marker_label
        observation.position.x = 10.0
        observation.position.y = -2.0
        observation.position.z = 0.0
        observation.confidence = 0.9
        observation.source = "mock_aruco"
        observation.yaw_rad = 0.0
        self.pub.publish(observation)


def main() -> None:
    rospy.init_node("aruco_node")
    ArucoNode()
    rospy.spin()


if __name__ == "__main__":
    main()
