import os
from copy import deepcopy
from typing import List

import torch 
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from .NN_actor_td3 import MLPNetworkActor_td3
from .NN_critic_td3 import MLPNetworkCritic_td3


class TD3:
    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device, action_bound,  chkpt_dir, chkpt_name):
        self.actor = MLPNetworkActor_td3(in_dim=obs_dim, out_dim=act_dim, hidden_dim = 64, action_bound=action_bound, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'actor_td3.pth')).to(device)
        self.critic = MLPNetworkCritic_td3(in_dim=global_obs_dim, out_dim=1, hidden_dim = 64, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'critic_td3.pth')).to(device)
        #优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = critic_lr)
        # 创建相对于的target网络
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        """
        使用 deepcopy 创建 target 网络是一个更好的选择，原因如下：
        初始化一致性：
            - deepcopy 确保 target 网络和原网络完全相同的初始参数
            - 重新创建网络可能因为随机初始化导致参数不一致
        """

    def actor_action(self, obs):
        # 如果是list，先合并为单个tensor
        # if isinstance(obs, list):
        #     obs = torch.cat(obs, dim=1)
        action = self.actor(obs)
        return action

    def actor_target_action(self, obs):
        # 如果是list，先合并为单个tensor
        if isinstance(obs, list):
            obs = torch.cat(obs, dim=1)
        action = self.actor_target(obs)
        return action

    def critic_qvalue(self, obs, action):
        """获取  critic网络  的Q值"""
        # 合并观测和动作
        if isinstance(obs, list) and isinstance(action, list):
            sa = torch.cat(list(obs) + list(action), dim=1)
        else:
            sa = torch.cat([obs, action], dim=1)
        q1, q2 = self.critic(sa)# 返回两个Q值
        return q1.squeeze(1), q2.squeeze(1)
    
    def critic_target_q(self, obs, action):
        """获取  critic目标网络  的Q值"""
        # 合并观测和动作
        if isinstance(obs, list) and isinstance(action, list):
            sa = torch.cat(list(obs) + list(action), dim=1)
        else:
            sa = torch.cat([obs, action], dim=1)
        q1, q2 = self.critic_target(sa)# 返回两个Q值
        return q1.squeeze(1), q2.squeeze(1)

    def critic_q1(self, obs, action):
        """只获取  critic网络的  第一个Q值  ，用于策略更新"""
        # 合并观测和动作
        if isinstance(obs, list) and isinstance(action, list):
            sa = torch.cat(list(obs) + list(action), dim=1)
        else:
            sa = torch.cat([obs, action], dim=1)  
        return self.critic.Q1(sa).squeeze(1) # 只返回Q1


    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        '''
            在较新版本的PyTorch中， clip_grad_norm 已被弃用，推荐使用 clip_grad_norm_
            clip_grad_norm_ 是 clip_grad_norm 的原地版本，不会创建新的张量，而是直接在输入张量上进行修改.
        '''
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 与clip_grad_norm的不同？
        self.actor_optimizer.step()
    
    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
