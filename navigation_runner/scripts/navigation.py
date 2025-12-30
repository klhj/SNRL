#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import math
import os
import tf.transformations

# ROS Messages
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Twist
from mavros_msgs.msg import State

# RL Related Imports
from ppo import PPO
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utils import vec_to_new_frame

class Navigation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.cfg.device
        
        # --- [修正 1] 恢复正确的参数配置 ---
        # Checkpoint 是基于 10.0 度 (36 beams) 和 4 层垂直扫描训练的
        self.cfg.sensor.lidar_hres = 10.0  
        self.lidar_hbeams = int(360 / self.cfg.sensor.lidar_hres) # = 36
        self.lidar_vbeams = 4 # 模型需要 4 层输入
        self.lidar_range = 5.0
        
        # 初始化容器
        self.scan_ranges = np.ones(self.lidar_hbeams) * self.lidar_range 
        
        self.robot_size = 0.3 
        
        # --- Navigation State ---
        self.goal = None
        self.goal_received = False
        self.target_dir = None
        self.has_action = False
        
        # --- Robot State ---
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_quat = [0, 0, 0, 1] 
        self.odom_received = False
        self.cmd_vel_world = np.zeros(3) 

        self.mavros_state = State()

        # --- Subscribers ---
        print("[nav-ros]: Initializing Subscribers...")
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        self.vel_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.vel_callback)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        
        # --- Publishers ---
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=1)
        self.cmd_vis_pub = rospy.Publisher("/rl_navigation/cmd", MarkerArray, queue_size=10)
        self.goal_vis_pub = rospy.Publisher("rl_navigation/goal", MarkerArray, queue_size=10)

        # --- Model ---
        self.policy = self.init_model()
        self.policy.eval()

        # --- Loop ---
        self.safety_stop = False 
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_callback)
        self.vis_timer = rospy.Timer(rospy.Duration(0.1), self.vis_callback)

        print(f"[nav-ros]: Node Ready. HRes: {self.cfg.sensor.lidar_hres}, Beams: {self.lidar_hbeams}, VBeams: {self.lidar_vbeams}")

    def init_model(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        
        # 使用计算好的维度构建 Spec
        # Shape: (1, 36, 4) -> 对应 (Batch, Beams, VBeams)
        lidar_shape = (1, self.lidar_hbeams, self.lidar_vbeams)
        
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.cfg.device), 
                    "lidar": UnboundedContinuousTensorSpec(lidar_shape, device=self.cfg.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.cfg.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.cfg.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.cfg.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.cfg.device), 
            })
        }).expand(1, action_dim).to(self.cfg.device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.cfg.device)

        curr_path = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(curr_path, "ckpts", "navrl_checkpoint.pt")
        
        if os.path.exists(ckpt_path):
            print(f"[nav-ros]: Loading model from {ckpt_path}")
            # 使用 strict=True 确保完全匹配
            policy.load_state_dict(torch.load(ckpt_path, map_location=self.cfg.device), strict=True)
            print("[nav-ros]: Model Loaded Successfully!")
        else:
            rospy.logerr(f"[nav-ros]: Checkpoint NOT found at {ckpt_path}")
            
        return policy

    def pose_callback(self, msg):
        self.current_pos[0] = msg.pose.position.x
        self.current_pos[1] = msg.pose.position.y
        self.current_pos[2] = msg.pose.position.z
        self.current_quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        self.odom_received = True

    def vel_callback(self, msg):
        self.current_vel[0] = msg.twist.linear.x
        self.current_vel[1] = msg.twist.linear.y
        self.current_vel[2] = msg.twist.linear.z

    def state_callback(self, msg):
        self.mavros_state = msg

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = self.lidar_range
        ranges[np.isnan(ranges)] = self.lidar_range
        ranges = np.clip(ranges, 0, self.lidar_range)
        
        full_scan = np.ones(self.lidar_hbeams) * self.lidar_range
        
        angle_curr = msg.angle_min
        angle_inc = msg.angle_increment
        rad_per_idx = np.deg2rad(self.cfg.sensor.lidar_hres) # 10.0 度

        for r in ranges:
            if r < self.lidar_range:
                # 简单的角度映射
                idx = int(angle_curr / rad_per_idx)
                if abs(idx) < (self.lidar_hbeams // 2):
                    # 注意：数组索引处理
                    array_idx = idx if idx >= 0 else self.lidar_hbeams + idx
                    if 0 <= array_idx < self.lidar_hbeams:
                        full_scan[array_idx] = min(full_scan[array_idx], r)
            angle_curr += angle_inc

        self.scan_ranges = full_scan

    def goal_callback(self, goal):
        if not self.odom_received:
            return
        self.goal = goal
        self.goal.pose.position.z = max(0.8, self.current_pos[2])
        self.goal_received = True
        print(f"[nav-ros]: New Goal: ({self.goal.pose.position.x:.2f}, {self.goal.pose.position.y:.2f})")

    def control_callback(self, event):
        if not self.odom_received or not self.goal_received:
            return
        
        if self.safety_stop:
            self.vel_pub.publish(Twist())
            return

        curr_pos_torch = torch.tensor(self.current_pos, device=self.device, dtype=torch.float)
        goal_pos_torch = torch.tensor([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z], device=self.device, dtype=torch.float)
        
        if torch.norm(curr_pos_torch[:2] - goal_pos_torch[:2]) < 0.3:
            print("[nav-ros]: Reached Goal!")
            self.vel_pub.publish(Twist())
            self.goal_received = False
            return

        self.target_dir = goal_pos_torch - curr_pos_torch
        vel_world = torch.tensor(self.current_vel, device=self.device, dtype=torch.float)

        cmd_vel_world = self.get_action(curr_pos_torch, vel_world, goal_pos_torch)
        cmd_vel_world = cmd_vel_world.squeeze(0).squeeze(0).detach().cpu().numpy()
        self.cmd_vel_world = cmd_vel_world
        
        final_cmd = Twist()
        max_speed = 0.8 
        speed_norm = np.linalg.norm(cmd_vel_world)
        if speed_norm > max_speed:
            cmd_vel_world = cmd_vel_world / speed_norm * max_speed
        
        final_cmd.linear.x = cmd_vel_world[0]
        final_cmd.linear.y = cmd_vel_world[1]
        final_cmd.linear.z = cmd_vel_world[2]
        
        desired_yaw = math.atan2(self.target_dir[1].item(), self.target_dir[0].item())
        curr_yaw = tf.transformations.euler_from_quaternion(self.current_quat)[2]
        yaw_err = desired_yaw - curr_yaw
        if yaw_err > math.pi: yaw_err -= 2*math.pi
        if yaw_err < -math.pi: yaw_err += 2*math.pi
        final_cmd.angular.z = 1.0 * yaw_err 

        # 打印调试信息
        print(f"[DEBUG] VEL: x={final_cmd.linear.x:.2f}, y={final_cmd.linear.y:.2f}, yaw={final_cmd.angular.z:.2f}")
        self.vel_pub.publish(final_cmd)
        self.has_action = True

    def get_action(self, pos, vel, goal):
        rpos = goal - pos
        distance = rpos.norm(dim=-1, keepdim=True)
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)

        target_dir_2d = self.target_dir.clone()
        target_dir_2d[2] = 0.
        target_dir_2d = target_dir_2d.unsqueeze(0)

        rpos_clipped = rpos / distance.clamp(1e-6)
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d).squeeze(0).squeeze(0)
        vel_g = vec_to_new_frame(vel, target_dir_2d).squeeze(0).squeeze(0)
        
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).unsqueeze(0)

        # --- [修正 2] 激光数据重构 ---
        lidar_tensor = torch.tensor(self.scan_ranges, device=self.device, dtype=torch.float)
        lidar_tensor = self.lidar_range - lidar_tensor
        lidar_tensor = lidar_tensor.clamp(min=0.0)
        
        # 将 1D 数组 (36,) 扩展为 (1, 1, 36, 1)
        lidar_tensor = lidar_tensor.reshape(1, 1, self.lidar_hbeams, 1)
        
        # 关键一步：复制数据以填充 4 个垂直层
        # expand 只是视图，repeat 是复制数据。为了安全用 repeat。
        # 目标: (1, 1, 36, 4)
        lidar_tensor = lidar_tensor.repeat(1, 1, 1, self.lidar_vbeams)

        dyn_obs_states = torch.zeros((1, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10), device=self.device)

        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state,
                    "lidar": lidar_tensor,
                    "direction": target_dir_2d,
                    "dynamic_obstacle": dyn_obs_states
                }, batch_size=[1])
            }, batch_size=[1])
        }, batch_size=[1])

        with set_exploration_type(ExplorationType.MEAN):
            output = self.policy(obs)
        return output["agents", "action"]

    def vis_callback(self, event):
        if self.goal_received and self.has_action:
            c_msg = MarkerArray()
            cm = Marker()
            cm.header.frame_id = "map"
            cm.header.stamp = rospy.Time.now()
            cm.ns = "cmd_vel"
            cm.id = 0
            cm.type = Marker.ARROW
            cm.action = Marker.ADD
            p_start = Point(*self.current_pos)
            cm.points.append(p_start)
            p_end = Point()
            p_end.x = self.current_pos[0] + self.cmd_vel_world[0]
            p_end.y = self.current_pos[1] + self.cmd_vel_world[1]
            p_end.z = self.current_pos[2] + self.cmd_vel_world[2]
            cm.points.append(p_end)
            cm.scale.x = 0.05; cm.scale.y = 0.1; cm.scale.z = 0.1
            cm.color.a = 1.0; cm.color.r = 0.0; cm.color.g = 1.0; cm.color.b = 0.0
            c_msg.markers.append(cm)
            self.cmd_vis_pub.publish(c_msg)