import os

import numpy as np
import torch
import torch.nn.functional as F

from .TD3_agent import TD3
from .buffer import BUFFER

class MATD3:
    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, action_bound, tau, _chkpt_dir, _device = 'cpu', _model_timestamp = None):
        self.device = _device
        self.model_timestamp = _model_timestamp
        # 状态（全局观测）与所有智能体动作维度的和 即critic网络的输入维度  dim_info =  [obs_dim, act_dim]
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())

        # 创建智能体与buffer，每个智能体有自己的buffer, actor, critic
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = TD3(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device, action_bound[agent_id], chkpt_name = (agent_id + '_'), chkpt_dir = _chkpt_dir) # 每一个智能体都是一个TD3_agent智能体
            self.buffers[agent_id] = BUFFER(capacity, obs_dim, act_dim, self.device)# buffer均只是存储自己的观测与动作
        
        self.dim_info = dim_info
        self.batch_size = batch_size
        self.tau =tau

        # TD3 特有的参数
        self.clip_double = True  # 是否使用双Q网络，即TD3算法，否则使用MADDPG算法
        self.regular = True # 是否使用正则化
        self.policy_noise = 0.1  # 目标策略平滑的噪声参数
        self.noise_clip = 0.5    # 噪声裁剪范围
        self.policy_freq = 3     # 策略延迟更新频率
        self.policy_noise_scale = 1
        self.policy_noise_init_scale = None  #若不设置衰减，则设置成None
        self.total_it = 0        # 记录更新次数

        self.noise_std = 0.1
        self.noise_mean = 0
        self.decay_rate = 0.999999
        self.min_noise = 0.01
        self.noise_scale = 1.0


    def add(self, obs, action, reward, next_obs, done):
        #NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)
    
    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # 获取总的转换数量，这些缓冲区应该具有相同数量的转换
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size = batch_size, replace = False)
        # 在MATD3中，我们需要所有智能体的观测和动作
        # 但在计算中只需要当前智能体的奖励和完成标志
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            # TD3特性：添加噪声以实现目标策略平滑
            if self.policy_noise is not None:
                noise = ( torch.randn_like(act[agent_id]) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
                next_act[agent_id] = (self.agents[agent_id].actor_target_action(next_obs[agent_id] ) + noise).clamp(-1.0, 1.0)
            else:
                next_act[agent_id] = self.agents[agent_id].actor_target_action(next_obs[agent_id] )                
        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs, explore=False, total_step=0, noise_type='gaussian'):
        """选择动作，用于与环境交互"""
        actions = {}
        for agent_id, agent_obs in obs.items():
            agent_obs = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
            action = self.agents[agent_id].actor_action(agent_obs)
            # 训练模式下添加噪声以增强探索
            if explore:
                action_dim = self.dim_info[agent_id][1]
                if noise_type == 'decay':
                    # 衰减噪声
                    noise_scale = max(self.min_noise, self.noise_scale * (self.decay_rate ** total_step))
                    noise = torch.randn_like(action) * noise_scale
                elif noise_type == "gaussian":
                    # 高斯噪声
                    noise = torch.randn_like(action) * self.noise_std + self.noise_mean
                    noise = torch.clamp(noise, -0.2, 0.2)
                elif noise_type == "uniform":
                    # 均匀分布噪声
                    noise = (torch.rand_like(action) * 2 - 1) * 0.1
                else:
                    raise ValueError(f"不支持的噪声类型: {noise_type}")
                
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)  # 假设动作范围是[-1,1]
        
            actions[agent_id] = action.cpu().data.numpy().flatten()  # TODO: 这里可能需要调整
        return actions

    def learn(self, batch_size, gamma):
        """MATD3学习过程"""
        self.total_it += 1
        # # 检查是否有足够的样本
        # agent_id = list(self.buffers.keys())[0]
        # if len(self.buffers[agent_id]) < batch_size:
        #     return
        # TD3算法核心部分
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(self.batch_size) # 采样经验，注意，采样出来的经验 next_act 带噪声
        
            if self.clip_double:  # 截断？
                next_target_q1, next_target_q2 = agent.critic_target_q(list(next_obs.values()), list(next_act.values()))
                next_target_Q = torch.min(next_target_q1, next_target_q2)
            else:
                next_target_Q = agent.critic_target_q(list(next_obs.values()), list(next_act.values()))
            # 先更新critic
            target_Q = reward[agent_id] + gamma * next_target_Q * (1 - done[agent_id]) 
            if self.clip_double:  # 截断？
                current_q1, current_q2 = agent.critic_qvalue(list(obs.values()), list(act.values()))
                # 确保维度匹配 - 添加调试信息
                # print(f"current_q1 shape: {current_q1.shape}, target_Q shape: {target_Q.shape}")
                critic_loss = F.mse_loss(current_q1, target_Q.detach()) + F.mse_loss(current_q2, target_Q.detach())  # TD3:两个loss加起来了 reduction = 'mean'
            else:
                current_Q = agent.critic_qvalue(list(obs.values()), list(act.values()))
                critic_loss = F.mse_loss(current_Q, target_Q.detach()) #MADDPG  reduction = 'mean'
            agent.update_critic(critic_loss)
            
            # 延迟更新actor
            if self.total_it % self.policy_freq == 0:  # 更新频率
                # 计算当前Q值
                action = agent.actor_action(obs[agent_id])
                act[agent_id] = action
                if self.clip_double:  # 截断？
                    actor_loss = -agent.critic_q1(list(obs.values()), list(act.values())).mean()  #TD3
                else:
                    actor_loss = -agent.critic_qvalue(list(obs.values()), list(act.values())).mean() # MADDPG
                if self.regular: # 和DDPG中的weight_decay一样原理
                    actor_loss += (action**2).mean() * 1e-3
                agent.update_actor(actor_loss)

        if self.total_it % self.policy_freq == 0:  # 目标网络更新频率
            self.update_target()


    def update_target(self):
        def soft_update(from_network, to_network): # 软更新
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau """
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(self.tau * from_p.data + (1.0 - self.tau) * to_p.data)
        for agent in self.agents.values():
            soft_update(agent.actor, agent.actor_target)  #体现使用嵌套函数的作用！ 易于维护和使用
            soft_update(agent.critic, agent.critic_target)


    @classmethod
    def load( cls, dim_info, file):
        """ init maddpg using the model saved in `file` """
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file, map_location=instance.device)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
    
    def save_model(self, timestamp = True):
        _timestamp = timestamp
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.save_checkpoint(is_target = False, timestamp = _timestamp)
            self.agents[agent_id].actor_target.save_checkpoint(is_target = True, timestamp = _timestamp)
            self.agents[agent_id].critic.save_checkpoint(is_target = False, timestamp = _timestamp)
            self.agents[agent_id].critic_target.save_checkpoint(is_target = True, timestamp = _timestamp)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组


    def load_model(self):
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].actor_target.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)
            self.agents[agent_id].critic.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].critic_target.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组
  
