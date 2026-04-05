#!/usr/bin/env python3

import rospy
from std_msgs.msg import String


class InstructionGatewayNode:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/vln/input/instruction")
        self.output_topic = rospy.get_param("~output_topic", "/vln/mission/raw_instruction")
        self.pub = rospy.Publisher(self.output_topic, String, queue_size=10, latch=True)
        self.sub = rospy.Subscriber(self.input_topic, String, self._on_instruction, queue_size=10)

    def _on_instruction(self, msg: String) -> None:
        self.pub.publish(msg)


def main() -> None:
    rospy.init_node("instruction_gateway_node")
    InstructionGatewayNode()
    rospy.spin()


if __name__ == "__main__":
    main()

