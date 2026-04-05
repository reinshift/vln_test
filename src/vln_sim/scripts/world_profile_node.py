#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid


class WorldProfileNode:
    def __init__(self) -> None:
        self.output_topic = rospy.get_param("~output_topic", "/vln/world/occupancy")
        self.resolution = float(rospy.get_param("~resolution", 1.0))
        self.width = int(rospy.get_param("~width", 20))
        self.height = int(rospy.get_param("~height", 20))
        self.pub = rospy.Publisher(self.output_topic, OccupancyGrid, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(1.0), self._publish, oneshot=True)

    def _publish(self, _event) -> None:
        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = "map"
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = -float(self.width) * self.resolution / 2.0
        grid.info.origin.position.y = -float(self.height) * self.resolution / 2.0
        grid.data = [0] * (self.width * self.height)
        self.pub.publish(grid)


def main() -> None:
    rospy.init_node("world_profile_node")
    WorldProfileNode()
    rospy.spin()


if __name__ == "__main__":
    main()

