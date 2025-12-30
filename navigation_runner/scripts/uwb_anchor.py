#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ================= 你的真实实验室坐标 (来自截图) =================
ANCHORS = [
    [0.0,    0.0,   0.0],  # A0 (原点)
    [-0.214, 5.895, 0.0],  # A1 (Y轴方向，略有偏差)
    [5.628,  5.865, 0.0],  # A2 (对角线)
    [5.681,  0.0,   0.0]   # A3 (X轴方向)
]
# ===============================================================

class UWBVisualizer:
    def __init__(self):
        rospy.init_node('uwb_anchor_node')
        self.marker_pub = rospy.Publisher('/lab/anchors', MarkerArray, queue_size=1, latch=True)
        self.fence_pub = rospy.Publisher('/lab/fence', Marker, queue_size=1, latch=True)
        rospy.loginfo("UWB Static Map Loaded with Custom Coordinates!")
        self.publish_static_map()

    def publish_static_map(self):
        # 1. 画基站 (红框)
        marker_array = MarkerArray()
        for i, coord in enumerate(ANCHORS):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = "anchors"; marker.id = i
            marker.type = Marker.CUBE; marker.action = Marker.ADD
            marker.pose.position.x = coord[0]; marker.pose.position.y = coord[1]; marker.pose.position.z = 2.0 # 假设高度挂在2米
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.8
            marker_array.markers.append(marker)
            
            # 文字标签
            text = Marker()
            text.header.frame_id = "map"
            text.ns = "text"; text.id = i + 100
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.text = f"A{i}"
            text.pose.position.x = coord[0]; text.pose.position.y = coord[1]; text.pose.position.z = 2.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.5
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0
            marker_array.markers.append(text)
        self.marker_pub.publish(marker_array)

        # 2. 画围栏 (黄线)
        fence = Marker()
        fence.header.frame_id = "map"
        fence.ns = "fence"; fence.id = 999
        fence.type = Marker.LINE_STRIP; fence.action = Marker.ADD
        fence.pose.orientation.w = 1.0
        fence.scale.x = 0.05
        fence.color.r = 1.0; fence.color.g = 1.0; fence.color.b = 0.0; fence.color.a = 1.0
        for coord in ANCHORS: fence.points.append(Point(x=coord[0], y=coord[1], z=0))
        fence.points.append(Point(x=ANCHORS[0][0], y=ANCHORS[0][1], z=0))
        self.fence_pub.publish(fence)

if __name__ == '__main__':
    try:
        UWBVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
