#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped

def handle_pose(msg):
    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    # 读取当前时间
    t.header.stamp = rospy.Time.now()
    
    # 强制连接 map -> base_link
    t.header.frame_id = "map"
    t.child_frame_id = "base_link"

    # 填充位置
    t.transform.translation.x = msg.pose.position.x
    t.transform.translation.y = msg.pose.position.y
    t.transform.translation.z = msg.pose.position.z

    # 填充姿态
    t.transform.rotation.x = msg.pose.orientation.x
    t.transform.rotation.y = msg.pose.orientation.y
    t.transform.rotation.z = msg.pose.orientation.z
    t.transform.rotation.w = msg.pose.orientation.w

    # 广播 TF
    br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('px4_tf_broadcaster')
    # 订阅 MAVROS 的位置话题
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, handle_pose)
    rospy.spin()
