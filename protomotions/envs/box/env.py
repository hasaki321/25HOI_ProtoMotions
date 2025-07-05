import numpy as np
import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
import isaaclab.sim as sim_utils

# 我们继承自 Steering，这样可以复用它的大部分逻辑
from protomotions.envs.steering.env import Steering, compute_heading_reward, compute_heading_observations

# 导入仿真器和配置相关的库
# from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState
# from protomotions.simulator.base_simulator.scene_object import SceneObject, SceneObjectType
# from protomotions.utils.math import quat_apply
# from protomotions.utils.common import unnormalize_heights

# (你可能需要将这个文件放到 protomotions/envs/box/ 目录下)

class PushBox(Steering):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        # 1. 初始化父类 (Steering)
        # 这样我们就拥有了所有 steering 的功能和变量
        super().__init__(config=config, device=device, *args, **kwargs)

        # 2. 读取我们自己的 box 任务配置
        self._box_obs_size = self.config.box_params.obs_size
        self._goal_dist_thresh = self.config.box_params.goal_dist_thresh # 到达目标的距离阈值
        self.contact_reward = self.config.box_params.contact_reward     # 第一次接触箱子的奖励
        self.goal_reward = self.config.box_params.goal_reward           # 到达目标的奖励

        # 3. 创建与箱子和目标相关的缓冲区 (Buffers)
        # 箱子
        self._create_box_asset()
        self.box_obs = torch.zeros((self.num_envs, self._box_obs_size), device=device, dtype=torch.float)
        
        # 目标点
        self.goal_pos = torch.zeros((self.num_envs, 3), device=device, dtype=torch.float)

        # 任务阶段管理
        # phase 0: 接近箱子 (approaching)
        # phase 1: 推动箱子 (pushing)
        self.task_phase = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        # 用于标记是否是第一次接触
        self.first_contact_done = torch.zeros(self.num_envs, device=device, dtype=torch.bool)

        self.scene = self.simulator._scene


    def _create_box_asset(self):
        """使用 Isaac Lab 的原生方式创建箱子资产"""
        from isaaclab.assets import  RigidObjectCfg, RigidObject
        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box", # 通配符路径，为每个环境创建一个
            spawn=sim_utils.CuboidCfg(
                size=(0.4, 0.4, 0.4),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # 确保箱子是动态的，可以被推动
                    kinematic_enabled=False 
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.1)
            ),
            init_state=RigidObjectCfg.InitialStateCfg()
        )

        self.box = RigidObject(cfg=box_cfg)
        

    
    # --- 重载核心方法 ---
    def reset(self, env_ids=None):
        """扩展 reset 逻辑"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        if len(env_ids) > 0:
            # 在调用父类 reset 之前，先重置我们自己的任务状态
            self._reset_box_and_goal(env_ids)
            
        # 调用父类的 reset，它会处理机器人重置和基础任务重置
        return super().reset(env_ids)

    def _reset_box_and_goal(self, env_ids):
        """
        在每次重置时，随机化箱子和目标的位置
        这是您的规划第一点
        """
        n = len(env_ids)
        # 获取机器人当前的重生位置
        robot_pos = self.simulator.get_root_state(env_ids).root_pos
        
        # 1. 在机器人前方一定半径圆内随机生成箱子位置
        radius_to_box = 1.5
        angle = (torch.rand(n, device=self.device) * 2 - 1) * np.pi
        dist = torch.rand(n, device=self.device) * radius_to_box + 0.8 # 至少0.8米远
        box_offset = torch.stack([torch.cos(angle) * dist, torch.sin(angle) * dist], dim=-1)
        
        box_pos_xy = robot_pos[:, :2] + box_offset
        box_pos_z = self.terrain.get_ground_heights(box_pos_xy) + 0.2 # Z 设为地面高度+箱子半高
        box_pos = torch.cat([box_pos_xy, box_pos_z], dim=-1)
        
        # 2. 在箱子前方一定半径圆内随机生成目标位置
        radius_to_goal = 2.0
        goal_angle = (torch.rand(n, device=self.device) * 2 - 1) * 0.5 * np.pi # 在前方+/-90度内
        goal_dist = torch.rand(n, device=self.device) * radius_to_goal + 1.0 # 至少1.0米远
        goal_offset = torch.stack([torch.cos(goal_angle) * goal_dist, torch.sin(goal_angle) * goal_dist], dim=-1)
        
        goal_pos_xy = box_pos[:, :2] + goal_offset
        goal_pos_z = self.terrain.get_ground_heights(goal_pos_xy) # Z 设为地面高度
        self.goal_pos[env_ids] = torch.cat([goal_pos_xy, goal_pos_z], dim=-1)

        # 通过API重置仿真器中箱子的状态
        # self.box_object.set_state(pos=box_pos, rot=torch.tensor([1,0,0,0], device=self.device).repeat(n,1), env_ids=env_ids)
        box_root_state = self.box.data.root_state_w.clone()
        # 修改需要重置的环境的对应状态
        box_root_state[env_ids, 0:3] = box_pos # 设置位置
        box_root_state[env_ids, 3:7] = torch.tensor([1,0,0,0], device=self.device) # 设置姿态为默认
        box_root_state[env_ids, 7:13] = 0 # 清空速度
        
        # 将修改后的状态一次性写入
        self.box.write_root_state_to_sim(box_root_state)

        # 重置任务阶段和标记
        self.task_phase[env_ids] = 0
        self.first_contact_done[env_ids] = False


    def compute_observations(self, env_ids=None):
        """
        扩展观测空间
        这是您的规划第三点
        """
        # 1. 先调用父类方法，获取 steering 的观测
        super().compute_observations(env_ids)

        # 2. 计算我们自己的 box 观测
        if env_ids is None:
            # 全量更新
            robot_pos = self.simulator.get_root_state().root_pos
            robot_rot = self.simulator.get_root_state().root_rot
            box_pos = self.box_object.get_state().pos
            goal_pos = self.goal_pos
            task_phase = self.task_phase
        else:
            # 增量更新
            robot_pos = self.simulator.get_root_state(env_ids).root_pos
            robot_rot = self.simulator.get_root_state(env_ids).root_rot
            box_pos = self.box_object.get_state(env_ids).pos
            goal_pos = self.goal_pos[env_ids]
            task_phase = self.task_phase[env_ids]

        # 计算机器人与箱子的关系 (在机器人局部坐标系下)
        vec_robot_to_box = box_pos - robot_pos
        heading_rot_inv = torch_utils.calc_heading_quat_inv(robot_rot, to_torch=True)
        # local_vec_robot_to_box = quat_apply(heading_rot_inv, vec_robot_to_box)
        local_vec_robot_to_box = rotations.quat_rotate(heading_rot_inv, vec_robot_to_box, True)

        # 计算箱子与目标点的关系 (在世界坐标系下，也可以转到局部)
        vec_box_to_goal = goal_pos - box_pos
        
        # 将两段关系拼接成一个观测向量
        # 这里的维度需要和你配置文件里的 `box_params.obs_size` 匹配
        box_obs_tensor = torch.cat([local_vec_robot_to_box, vec_box_to_goal], dim=-1)
        
        # 根据任务阶段，决定 steering 的目标是什么
        # phase 0: 目标是箱子
        # phase 1: 目标是最终目标点
        target_dir_for_steering = torch.where(
            task_phase.unsqueeze(-1) == 0,
            vec_robot_to_box[:, :2], # 接近阶段，目标方向是朝向箱子
            vec_box_to_goal[:, :2]   # 推动阶段，目标方向是朝向终点
        )
        target_dir_for_steering = torch.nn.functional.normalize(target_dir_for_steering, p=2, dim=-1)
        
        # --- 关键修改：重写 steering 的目标 ---
        # 我们不再使用 steering 自己随机生成的目标，而是用我们任务相关的目标
        # 这就是将两个任务逻辑结合起来的核心
        target_speed_for_steering = torch.ones_like(self.task_phase, dtype=torch.float) * 1.5 # 设定一个恒定目标速度
        steering_obs_tensor = compute_heading_observations(robot_rot, target_dir_for_steering, target_speed_for_steering)
        
        # 更新观测缓冲区
        if env_ids is None:
            self.box_obs[:] = box_obs_tensor
            self.steering_obs[:] = steering_obs_tensor
        else:
            self.box_obs[env_ids] = box_obs_tensor
            self.steering_obs[env_ids] = steering_obs_tensor

    def get_obs(self):
        """扩展 get_obs，将 box_obs 加入到总观测字典中"""
        obs = super().get_obs()
        obs.update({"box": self.box_obs})
        return obs

    def post_physics_step(self):
        """
        扩展 post_physics_step，加入阶段转换逻辑
        这是您的规划第二点
        """
        super().post_physics_step() # 这会调用 check_update_task，但我们可以忽略它的效果，因为我们重写了steering的目标

        # 检查是否接触到了箱子
        # (这里用一个简单的距离判断代替真实的接触检测，真实接触检测需要从 self.simulator 获取)
        box_positions = self.box.data.root_pos_w # 获取所有箱子的世界坐标
        dist_robot_to_box = torch.norm(self.simulator.get_root_state().root_pos[:, :2] - box_positions[:, :2], dim=-1)
        # dist_robot_to_box = torch.norm(self.simulator.get_root_state().root_pos[:, :2] - self.box_object.get_state().pos[:, :2], dim=-1)
        contact_mask = (dist_robot_to_box < 0.6) & (~self.first_contact_done)
        
        # 如果接触了，并且是第一次
        if torch.any(contact_mask):
            contact_env_ids = contact_mask.nonzero(as_tuple=False).flatten()
            self.task_phase[contact_env_ids] = 1 # 进入推动阶段
            self.rew_buf[contact_env_ids] += self.contact_reward # 给予一次性接触奖励
            self.first_contact_done[contact_env_ids] = True # 标记已完成第一次接触

    def compute_reward(self):
        """
        完全重载 compute_reward，实现我们自己的两阶段奖励
        这是您的规划第二点
        """
        # 1. 计算 steering reward，这是我们的基础引导奖励
        # 我们需要自己调用 compute_heading_reward 函数
        robot_pos = self.simulator.get_root_state().root_pos
        prev_robot_pos = self._prev_root_pos # steering 基类里有这个变量
        
        # 根据任务阶段，决定 steering 的目标
        target_dir_for_steering = torch.where(
            self.task_phase.unsqueeze(-1) == 0,
            torch.nn.functional.normalize(self.box_object.get_state().pos[:, :2] - robot_pos[:, :2], p=2, dim=-1),
            torch.nn.functional.normalize(self.goal_pos[:, :2] - self.box_object.get_state().pos[:, :2], p=2, dim=-1)
        )
        target_speed_for_steering = torch.ones_like(self.task_phase, dtype=torch.float) * 1.5
        
        steering_reward = compute_heading_reward(robot_pos, prev_robot_pos, target_dir_for_steering, target_speed_for_steering, self.dt)

        # 2. 计算额外的任务奖励
        # 这里只计算一个到达目标的奖励，因为推动过程中的奖励已经由 steering_reward 提供了
        box_positions = self.box.data.root_pos_w
        dist_box_to_goal = torch.norm(box_positions[:, :2] - self.goal_pos[:, :2], dim=-1)
        # dist_box_to_goal = torch.norm(self.box_object.get_state().pos[:, :2] - self.goal_pos[:, :2], dim=-1)
        goal_reached_mask = (dist_box_to_goal < self._goal_dist_thresh)
        
        goal_reward = torch.zeros_like(self.rew_buf)
        goal_reward[goal_reached_mask] = self.goal_reward

        # 3. 组合总奖励
        self.rew_buf[:] = steering_reward * self.config.steering_params.reward_scale + goal_reward
        
        # 更新 _prev_root_pos
        self._prev_root_pos[:] = robot_pos.clone()


    def compute_reset(self):
        """扩展重置条件，如果到达目标点也重置"""
        super().compute_reset() # 先计算基础的重置（如摔倒）

        # 增加到达目标的重置条件
        dist_box_to_goal = torch.norm(self.box_object.get_state().pos[:, :2] - self.goal_pos[:, :2], dim=-1)
        goal_reached_mask = (dist_box_to_goal < self._goal_dist_thresh)
        
        self.reset_buf[goal_reached_mask] = 1
        self.terminate_buf[goal_reached_mask] = 1 # 表示成功终止