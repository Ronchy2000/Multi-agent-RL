import os
from copy import deepcopy
from typing import List

import torch 
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam

class DDPG():
    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device, action_bound,  chkpt_dir, chkpt_name):
        self.actor = MLPNetworkActor(in_dim=obs_dim, out_dim=act_dim, hidden_dim = 64, action_bound=action_bound, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'actor.pth')).to(device)
        self.critic = MLPNetworkCritic(in_dim=global_obs_dim, out_dim=1, hidden_dim = 64, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'critic.pth')).to(device)
        #优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = critic_lr)
        # 创建相对于的target网络
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
        nn.utils.clip_grad_norm(self.actor.parameters(), 0.5)  # TODO
        self.actor_optimizer.step()
    
    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)  # TODO
        self.critic_optimizer.step()

"""
self.target_critic = CriticNetwork(*, *, 
                                chkpt_dir=chkpt_dir,
                                name=self.agent_name+'_target_critic')
"""
class MLPNetworkCritic(nn.Module):
    def __init__(self, chkpt_name,  chkpt_dir, in_dim, out_dim, hidden_dim = 64, non_linear = nn.ReLU()):
        super(MLPNetworkCritic, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, chkpt_name)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        '''init patameters of the module'''
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = gain)  #使用了 Xavier 均匀分布初始化（也叫 Glorot 初始化）
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.net(x)
    
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

class MLPNetworkActor(nn.Module):
    def __init__(self,chkpt_name,  chkpt_dir, in_dim, out_dim, action_bound, hidden_dim = 64, non_linear = nn.ReLU()):
        super(MLPNetworkActor, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, chkpt_name)

        # different ,为什么要保持这两个信息？
        self.out_dim = out_dim
        self.action_bound = action_bound

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        '''init patameters of the module'''
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = gain)  #使用了 Xavier 均匀分布初始化（也叫 Glorot 初始化）
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.net(x)
        logi = x
        a_min = self.action_bound[0]
        a_max = self.action_bound[1]
        ''' 这三行为什么要这么处理？ 引入了bias项干嘛'''
        k = torch.tensor( (a_max - a_min) /2)
        bias = torch.tensor( (a_max + a_min) /2 )
        action = k * torch.tanh(x) + bias
        return action, logi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))