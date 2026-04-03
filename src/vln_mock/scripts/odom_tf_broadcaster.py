#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
import tf
import tf.transformations as tfs

class OdomTFBroadcaster:
    def __init__(self):
        rospy.init_node('odom_tf_broadcaster', anonymous=True)
        self.br = tf.TransformBroadcaster()
        self.odometry_topic = rospy.get_param('~odometry_topic', '/magv/odometry/gt')
        self.default_child = rospy.get_param('~default_child_frame_id', 'magv/base_link')
        self.alias_child = rospy.get_param('~alias_child_frame_id', '')  # optional alias
        self.sub = rospy.Subscriber(self.odometry_topic, Odometry, self.odom_cb, queue_size=10)
        rospy.loginfo(
            'odom_tf_broadcaster started. odom_topic=%s default_child=%s alias_child=%s',
            self.odometry_topic,
            self.default_child,
            self.alias_child,
        )

    def odom_cb(self, msg: Odometry):
        parent = msg.header.frame_id or 'map'
        child = msg.child_frame_id or self.default_child
        # broadcast parent->child
        self.br.sendTransform(
            (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
            msg.header.stamp,
            child,
            parent,
        )
        # optional alias: also publish parent->alias_child with same transform
        if self.alias_child:
            self.br.sendTransform(
                (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
                (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
                msg.header.stamp,
                self.alias_child,
                parent,
            )

if __name__ == '__main__':
    try:
        OdomTFBroadcaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
