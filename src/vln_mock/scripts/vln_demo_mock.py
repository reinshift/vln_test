#!/usr/bin/env python

import rospy
import math
import io
import numpy as np
import tf
from PIL import Image

import sensor_msgs.point_cloud2 as pc2

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from magv_vln_msgs.msg import PositionCommand
from sensor_msgs.msg import PointCloud2, CompressedImage
from geometry_msgs.msg import Twist

class VLNDemoMock:
    def __init__(self):
        # 初始化节点
        rospy.init_node('vln_demo_mock', anonymous=True)
        
        # 创建发布者，发布到/status话题
        self.status_pub = rospy.Publisher('/status', Int32, queue_size=10)
        self.target_pub_discrete = rospy.Publisher('/magv/planning/pos_cmd', PositionCommand, queue_size=10)
        self.target_pub_continuous = rospy.Publisher('/magv/omni_drive_controller/cmd_vel', Twist, queue_size=10)
        
        # 创建订阅者，订阅各类话题
        rospy.Subscriber('/instruction', String, self.instruction_callback)
        rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback)
        rospy.Subscriber('/magv/scan/3d', PointCloud2, self.pointcloud_callback)
        rospy.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage, self.image_callback)
        
        rospy.loginfo("VLN Demo Mock node started, waiting for instruction...")

    # 该示例发送目标消息，向车体坐标系x轴正方向移动1m
    def example_moving_forward(self):
        #构建目标消息
        goal1 = PositionCommand()
        goal1.position.x = self.odom_x + 1.0 * math.cos(self.odom_yaw)  # 目标x坐标
        goal1.position.y = self.odom_y + 1.0 * math.sin(self.odom_yaw) # 目标y坐标
        goal1.position.z = 0.0 
        goal1.velocity.x = 0.0
        goal1.velocity.y = 0.0
        goal1.velocity.z = 0.0
        goal1.yaw = self.odom_yaw + 0.0  #目标朝向
        goal1.yaw_dot = 0.0
        
        rospy.loginfo("Sending goal: x=%.2f, y=%.2f, theta=%.2f", 
                     goal1.position.x, goal1.position.y, goal1.yaw)
        
        self.target_pub_discrete.publish(goal1)  # 发送目标消息

    # 该示例发送目标消息，偏航角向左转90度
    def example_turn_left(self):
        #构建目标消息
        goal2 = PositionCommand()
        goal2.position.x = self.odom_x  # 目标x坐标
        goal2.position.y = self.odom_y  # 目标y坐标
        goal2.position.z = 0.0
        goal2.velocity.x = 0.0
        goal2.velocity.y = 0.0
        goal2.velocity.z = 0.0
        goal2.yaw = self.odom_yaw + math.pi / 2  #目标朝向
        goal2.yaw_dot = 0.0 

        rospy.loginfo("Sending goal: x=%.2f, y=%.2f, theta=%.2f", 
                     goal2.position.x, goal2.position.y, goal2.yaw)
        
        self.target_pub_discrete.publish(goal2)  # 发送目标消息
    
    # 该示例发送目标消息，设置magv速度为车体坐标系x方向1m/s，由于控制器设定，实际会以该速度行驶2s，2s之后停下，因此若需要持续行驶，需要周期性发送目标消息（不低于0.5Hz）
    def example_moving_forward_continuous(self):
        #构建目标消息
        goal3 = Twist()
        goal3.linear.x = 1.0  # x轴速度
        goal3.linear.y = 0.0  # y轴速度
        goal3.linear.z = 0.0  
        goal3.angular.x = 0.0  
        goal3.angular.y = 0.0  
        goal3.angular.z = 0.0  # 偏航角速度

        rospy.loginfo("Sending velocity: x=%.2f, y=%.2f, theta=%.2f", 
                     goal3.linear.x, goal3.linear.y, goal3.angular.z)

        self.target_pub_continuous.publish(goal3)  # 发送目标消息

    # 该示例发送目标消息，设置magv速度为偏航角速度为0.5rad/s，由于控制器设定，实际会以该速度行驶2s，2s之后停下，因此若需要持续行驶，需要周期性发送目标消息（不低于0.5Hz）
    def example_turn_left_continuous(self):
        #构建目标消息
        goal4 = Twist()
        goal4.linear.x = 0.0  # x轴速度
        goal4.linear.y = 0.0  # y轴速度
        goal4.linear.z = 0.0  
        goal4.angular.x = 0.0
        goal4.angular.y = 0.0
        goal4.angular.z = 0.5  # 偏航角速度

        rospy.loginfo("Sending velocity: x=%.2f, y=%.2f, theta=%.2f", 
                     goal4.linear.x, goal4.linear.y, goal4.angular.z)

        self.target_pub_continuous.publish(goal4)  # 发送目标消息
        
    def instruction_callback(self, msg):
        """
        Callback function for receiving instruction messages
        """
        rospy.loginfo("Received instruction: %s", msg.data)
        rospy.loginfo("Processing, waiting 2 seconds...")
        rospy.loginfo("Processing, waiting 2 seconds...")
        
        # 等待2秒
        rospy.sleep(2.0)

        self.example_moving_forward_continuous()

        rospy.sleep(3.0)

        self.example_turn_left_continuous()

        rospy.sleep(3.0)

        self.example_moving_forward()

        rospy.sleep(5.0)

        self.example_turn_left()

        rospy.sleep(5.0)

        # 发送状态为0的消息，表示任务结束
        status_msg = Int32()
        status_msg.data = 0
        self.status_pub.publish(status_msg)
        rospy.loginfo("Status sent: 0")
        

    def odometry_callback(self, msg):
        """
        Callback function for receiving odometry messages
        """
        #rospy.loginfo("Received odometry: %s", msg)
        #magv当前的位置信息保存在msg.pose.pose下（注意有两层pose）
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        # 提取四元数
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 转换为欧拉角
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        self.odom_yaw = yaw  # 提取偏航角

        #magv当前的速度信息保存在msg.twist.twist下（注意有两层twist）
        self.vel_x = msg.twist.twist.linear.x#世界坐标系X轴方向速度
        self.vel_y = msg.twist.twist.linear.y#世界坐标系Y轴方向速度
        self.vel_yaw = msg.twist.twist.angular.z#世界坐标系偏航角速度s

    #这里是一个输出全部点云数据的示例，仅作参考	
    #这里使用senser_msgs解析和操作 PointCloud2 数据的工具库，选手也可以自行选用其他库，如ros_numpy
    def pointcloud_callback(self, msg):
        """
        Callback function for receiving pointcloud messages
        """
        # 读取点云中的点，并滤除异常值
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # 计算点云数量
        pc_number = len(list(points))
        #rospy.loginfo("Received pointcloud of number: %d", pc_number)
        # 打印每个点
        for point in points:
            x, y, z = point
            # rospy.loginfo("Point: x=%f, y=%f, z=%f", x, y, z)

    #这里是一个输出接收图像的示例，仅作参考
    #这里使用numpy和Pillow库处理压缩图像
    def image_callback(self, msg):
        """
        Callback function for receiving compressed image messages using Pillow
        """
        # 1. 用Pillow解码为PIL.Image对象
        image = Image.open(io.BytesIO(msg.data))
        # 2. 转为numpy数组（格式为HWC，RGB）
        np_image = np.array(image)
        #rospy.loginfo("Received image of shape: %s", np_image.shape)
        # 例如：显示图片（调试时用，Pillow自带show方法）
        # image.show()

if __name__ == '__main__':
    try:
        vln_mock = VLNDemoMock()
        rospy.spin()  # 保持节点运行
    except rospy.ROSInterruptException:
        pass
