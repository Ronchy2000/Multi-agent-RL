import os

import numpy as np
import torch
import torch.nn.functional as F
from agents.DDPG_agent import DDPG
from agents.buffer import BUFFER

class MADDPG():
    # device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, action_bound, _chkpt_dir, _device = 'cpu', _model_timestamp = None):
        self.device = _device
        self.model_timestamp = _model_timestamp
        # 状态（全局观测）与所有智能体动作维度的和 即critic网络的输入维度  dim_info =  [obs_dim, act_dim]
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # 创建智能体与buffer，每个智能体有自己的buffer, actor, critic
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            # print("dim_info -> agent_id:",agent_id)
            # 每一个智能体都是一个DDPG智能体
            
            self.agents[agent_id] = DDPG(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device, action_bound[agent_id], chkpt_name = (agent_id + '_'), chkpt_dir = _chkpt_dir)
            # buffer均只是存储自己的观测与动作
            self.buffers[agent_id] = BUFFER(capacity, obs_dim, act_dim, self.device)
        self.dim_info = dim_info
        self.batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):
        #NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):  #返回值为True or False, 判断a是否为int类型，是，返回True。
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)
    
    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size = batch_size, replace = False)
        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id], _ = self.agents[agent_id].target_action(n_o)
        
        return obs, act, reward, next_obs, done, next_act
    
    def select_action(self, obs):
        action = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            a, _ = self.agents[agent].action(o)   # torch.Size([1, action_size])    #action函数：  action, logi = self.actor(obs)
            # NOTE that the output is a tensor, convert it to int before input to the environment
            action[agent] = a.squeeze(0).detach().cpu().numpy()
        return action
    # 更多解释-飞书链接：https://m6tsmtxj3r.feishu.cn/docx/Kb1vdqvBholiIUxcvYxcIcBcnEg?from=from_copylink   密码：6u2257#8
    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # upate critic
            critic_value = agent.critic_value( list(obs.values()), list(act.values()) )

            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value* (1-done[agent_id])
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction = 'mean')
            agent.update_critic(critic_loss)

            #update actor
            action, logits = agent.action(obs[agent_id], model_out = True)
            act[agent_id] = action
            actor_loss = - agent.critic_value( list(obs.values()), list(act.values()) ).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()  #这个是干嘛的？
            agent.update_actor(actor_loss + 1e-3 *actor_loss_pse)
    
    def update_target(self, tau): #  嵌套函数定义
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau """
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)  #体现使用嵌套函数的作用！ 易于维护和使用
            soft_update(agent.critic, agent.target_critic)

    @classmethod
    def load( cls, dim_info, file):
        """ init maddpg using the model saved in `file` """
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file, map_location=instance.device)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
    
    def save_model(self):
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.save_checkpoint(is_target = False, timestamp = True)
            self.agents[agent_id].target_actor.save_checkpoint(is_target = True, timestamp = True)
            self.agents[agent_id].critic.save_checkpoint(is_target = False, timestamp = True)
            self.agents[agent_id].target_critic.save_checkpoint(is_target = True, timestamp = True)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组


    def load_model(self):
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].target_actor.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)
            self.agents[agent_id].critic.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].target_critic.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组
  
