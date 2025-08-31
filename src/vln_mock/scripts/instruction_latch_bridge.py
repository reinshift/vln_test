#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String as StringMsg

class InstructionLatchBridge(object):
    def __init__(self):
        rospy.init_node('instruction_latch_bridge', anonymous=True)
        rospy.loginfo("[instruction_latch_bridge] starting...")

        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/instruction')
        self.output_topic = rospy.get_param('~output_topic', '/instruction')
        self.wait_timeout = rospy.get_param('~timeout_sec', 0)  # 0 means wait forever

        self._latched_pub = None
        self._got_first = False

        # Subscribe early to catch the very first instruction
        self._sub = rospy.Subscriber(self.input_topic, StringMsg, self._on_msg, queue_size=10)

        if self.wait_timeout > 0:
            rospy.Timer(rospy.Duration(self.wait_timeout), self._on_timeout, oneshot=True)

        rospy.loginfo("[instruction_latch_bridge] listening on %s, will republish latched to %s"
                      % (self.input_topic, self.output_topic))

    def _ensure_latched_pub(self):
        if self._latched_pub is None:
            # Create latched publisher on demand
            self._latched_pub = rospy.Publisher(self.output_topic, StringMsg, queue_size=1, latch=True)
            # Give ROS a moment to register publisher
            rospy.sleep(0.05)

    def _on_msg(self, msg):
        if self._got_first:
            return
        self._got_first = True

        rospy.loginfo("[instruction_latch_bridge] captured first instruction: %s" % msg.data)

        # Create latched publisher and publish the captured message
        self._ensure_latched_pub()
        try:
            self._latched_pub.publish(msg)
            rospy.loginfo("[instruction_latch_bridge] re-published latched instruction on %s" % self.output_topic)
        except Exception as e:
            rospy.logerr("[instruction_latch_bridge] failed to publish latched instruction: %s" % e)

        # Unregister subscriber to avoid echo loop and unnecessary processing
        try:
            self._sub.unregister()
            rospy.loginfo("[instruction_latch_bridge] subscriber unregistered to avoid loop")
        except Exception as e:
            rospy.logwarn("[instruction_latch_bridge] failed to unregister subscriber: %s" % e)

    def _on_timeout(self, _event):
        if not self._got_first:
            rospy.logwarn("[instruction_latch_bridge] timeout reached without receiving any instruction")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = InstructionLatchBridge()
        node.spin()
    except rospy.ROSInterruptException:
        pass