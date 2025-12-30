#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped, Twist
from cv_bridge import CvBridge, CvBridgeError

class HUDVisualizer:
    def __init__(self):
        rospy.init_node('hud_visualizer', anonymous=True)

        # === 核心配置 ===
        self.image_topic = "/camera/color/image_raw"
        self.cmd_topic = "/mavros/setpoint_velocity/cmd_vel_unstamped" 
        self.pub_topic = "/camera/hud_view"

        # === 初始化 ===
        self.bridge = CvBridge()
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ang_z = 0.0

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        try:
            self.cmd_sub = rospy.Subscriber(self.cmd_topic, Twist, self.cmd_callback)
        except:
            pass
            
        self.image_pub = rospy.Publisher(self.pub_topic, Image, queue_size=1)
        rospy.loginfo("HUD Visualizer V4 (Final Polish) Started!")

    def cmd_callback(self, msg):
        if hasattr(msg, 'twist'):
            vel = msg.twist
        else:
            vel = msg 
        self.vx = vel.linear.x
        self.vy = vel.linear.y
        self.vz = vel.linear.z
        self.ang_z = vel.angular.z

    def draw_outlined_text(self, img, text, pos, font_scale, text_color, thickness=1, outline_bonus=2):
        """
        绘制带描边的文字。
        outline_bonus: 描边比内芯多粗。设小一点会显瘦。
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 1. 黑色描边
        # 这里的 outline_thick 决定了外框多“肥”
        outline_thick = thickness + outline_bonus
        cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), outline_thick, cv2.LINE_AA)
        
        # 2. 彩色内芯
        cv2.putText(img, text, pos, font, font_scale, text_color, thickness, cv2.LINE_AA)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        rows, cols, _ = cv_image.shape
        center_x = int(cols / 2)
        
        # === 1. 绘制箭头 (长度减半) ===
        start_pt = (center_x, int(rows * 0.8))
        
        # 修改：比例从 150 改为 75，缩短一半
        scale_linear = 75 
        scale_lateral = 75
        
        end_y = int(start_pt[1] - (self.vx * scale_linear))
        end_x = int(start_pt[0] - (self.vy * scale_lateral))

        end_x = max(10, min(end_x, cols-10))
        end_y = max(10, min(end_y, rows-10))

        if abs(self.vx) > 0.05 or abs(self.vy) > 0.05:
            # 修改：箭头粗细从 4 改为 3，配合短箭头更协调
            cv2.arrowedLine(cv_image, start_pt, (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)

        # === 2. 绘制文字 (位置极致优化) ===
        font = cv2.FONT_HERSHEY_SIMPLEX

        # A. 左上角：Onboard Camera View 
        # 修改：位置从 (10, 30) -> (5, 25)，极限贴边
        # 字体缩小一点点 (0.8 -> 0.7) 显得更精致
        cv2.putText(cv_image, "Onboard Camera View", (5, 25), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA) 
        cv2.putText(cv_image, "Onboard Camera View", (5, 25), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA) 

        # B. 左下角：Action Output
        # 修改：位置从 (10, rows-50) -> (5, rows-40)，往左下角挤
        self.draw_outlined_text(cv_image, "Action Output", (5, rows - 40), 0.8, (255, 255, 255), thickness=1, outline_bonus=2)

        # C. 底部正中：数据 vx, vy, vz (极简瘦身版)
        data_str = f"vx: {self.vx:.2f}  vy: {self.vy:.2f}  vz: {self.vz:.2f}"
        
        # 计算居中
        # 修改：字体大小 0.65 -> 0.6
        font_scale_data = 0.6
        text_size = cv2.getTextSize(data_str, font, font_scale_data, 1)[0]
        text_x = int((cols - text_size[0]) / 2)
        # 修改：位置极限靠底 (rows-10)
        text_y = rows - 10 
        
        # 修改：outline_bonus=1 (只比内芯粗1个像素)，这是让字看起来“瘦”的关键！
        # 颜色换成了稍淡一点的紫色/粉色，视觉上不那么胀
        self.draw_outlined_text(cv_image, data_str, (text_x, text_y), font_scale_data, (180, 180, 255), thickness=1, outline_bonus=1)

        # === 发布 ===
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    try:
        HUDVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass