import os
from copy import deepcopy
from typing import List

import torch 
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from agents.NN_actor import MLPNetworkActor
from agents.NN_critic import MLPNetworkCritic

class DDPG():
    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device, action_bound,  chkpt_dir, chkpt_name):
        self.actor = MLPNetworkActor(in_dim=obs_dim, out_dim=act_dim, hidden_dim = 64, action_bound=action_bound, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'actor.pth')).to(device)
        self.critic = MLPNetworkCritic(in_dim=global_obs_dim, out_dim=1, hidden_dim = 64, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'critic.pth')).to(device)
        #优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = critic_lr)
        # 创建相对于的target网络
        """
        使用 deepcopy 创建 target 网络是一个更好的选择，原因如下：
        初始化一致性：
            - deepcopy 确保 target 网络和原网络完全相同的初始参数
            - 重新创建网络可能因为随机初始化导致参数不一致
        """
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def action(self, obs, model_out = False):
        # 其中没有用到logi, 接受其返回值第二项为 '_' 具体地:  a, _ = self.agents[agent].action(o) 
        action, logi = self.actor(obs)
        return action, logi

    def target_action(self,obs):
        action, logi = self.target_actor(obs)
        return action, logi
    
    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):  # 包含Tensor对象的列表
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length
    
    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length
    
    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # clip_grad_norm_ ：带有下划线后缀，表示这是一个就地操作，会直接修改传入的参数梯度。
        self.actor_optimizer.step()
    
    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # clip_grad_norm_ ：带有下划线后缀，表示这是一个就地操作，会直接修改传入的参数梯度。
        self.critic_optimizer.step()
