# -*- coding: utf-8 -*-
#
# @Time : 2024-10-25 15:22:13
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : maddpg_agent_v1.py
# @Software: PyCharm
# @Description: None

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device: {}".format(device))
class ReplayBuffer():
    def __init__(self, capacity, obs_dim, state_dim, action_dim, batch_size):
        self.capacity = capacity
        self.obs_cap = np.empty((self.capacity, obs_dim))
        self.next_obs_cap = np.empty((self.capacity, obs_dim))
        self.state_cap = np.empty((self.capacity, state_dim))
        self.next_state_cap = np.empty((self.capacity, state_dim))
        self.action_cap = np.empty((self.capacity, action_dim))
        self.reward_cap = np.empty((self.capacity, 1))
        self.done_cap = np.empty((self.capacity, 1), dtype = bool)
        self.batch_size = batch_size
        self.current = 0
    def add_memory(self, obs, next_obs, state, next_state, action, reward, done):
        self.obs_cap[self.current] = obs
        self.next_obs_cap[self.current] = next_obs
        self.state_cap[self.current] = state
        self.next_state_cap[self.current] = next_state
        self.action_cap[self.current] = action
        self.reward_cap[self.current] = reward
        self.done_cap[self.current] = done

        self.current = (self.current + 1) % self.capacity #  memory空间放满后，又从第一个开始存放。

    def sample(self, idxed):
        obs = self.obs_cap[idxed]
        next_obs = self.next_obs_cap[idxed]
        state = self.state_cap[idxed]
        next_state = self.next_state_cap[idxed]
        action = self.action_cap[idxed]
        reward = self.reward_cap[idxed]
        done = self.done_cap[idxed]
        return obs, next_obs, state, next_state, action, reward, done


class Actor(nn.Module):
    def __init__(self, lr_actor, input_dim, fc1_dim, fc2_dim, action_dim):
        super(Actor, self).__init__()
        # self.fc1 = nn.Linear(input_dim, fc1_dim)
        # self.fc2 = nn.linear(fc1_dim, fc2_dim)
        # self.pi = nn.Linear(fc2_dim, action_dim)

        # 含有两个隐藏层的模型
        self.model_actor = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(), # sequential中的ReLU不需要参数
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(), # sequential中的ReLU不需要参数
            nn.Linear(fc2_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        pi = self.model_actor(state)
        mu = torch.softmax(pi, dim = 1)
        return mu
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class Critic(nn.Module):
    def __init__(self, lr_critic, input_dim, fc1_dim, fc2_dim, num_agent, action_dim):
        super(Critic, self).__init__()
        # 含有两个隐藏层的模型
        self.model_critic = nn.Sequential(
            nn.Linear(input_dim + num_agent * action_dim, fc1_dim),
            nn.ReLU(),  # sequential中的ReLU不需要参数
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),  # sequential中的ReLU不需要参数
            nn.Linear(fc2_dim, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr_critic)

    def forward(self, state, action):
        # x = torch.cat([state, action])
        x = torch.cat([state.to(device), action.to(device)], dim=1)
        q = self.model_critic(x)
        return q
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class Agent():
    def __init__(self, memory_size, obs_dim, state_dim, n_agent, action_dim,
                 alpha, beta, fc1_dim, fc2_dim, gamma, tau, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(lr_actor = alpha, input_dim = obs_dim, fc1_dim = fc1_dim, fc2_dim= fc2_dim, \
                           action_dim = action_dim).to(device)

        self.target_actor = Actor(lr_actor=alpha, input_dim=obs_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim, \
                           action_dim=action_dim).to(device)

        self.critic = Critic(lr_critic = beta, input_dim = state_dim, fc1_dim = fc1_dim, fc2_dim = fc2_dim, \
                             num_agent = n_agent, action_dim = action_dim).to(device)
        self.target_critic = Critic(lr_critic=beta, input_dim=state_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim, \
                             num_agent=n_agent, action_dim=action_dim).to(device)
        self.replay_buffer = ReplayBuffer(capacity=memory_size, obs_dim= obs_dim, state_dim = state_dim,\
                                          action_dim = action_dim, batch_size = batch_size)
    def get_action(self, obs):
        single_obs = torch.tensor(data = obs, dtype = torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device) * 0.2  #缩放一下噪声
        single_action = torch.clamp(input = single_action+noise, min = 0.0, max = 1.0)  #限制在0-1之间
        return single_action.detach().cpu().numpy()[0]

    def save_model(self, filename):
        self.actor.save_checkpoint(filename)
        self.target_actor.save_checkpoint(filename)
        self.critic.save_checkpoint(filename)
        self.target_critic.save_checkpoint(filename)
    def load_model(self, filename):
        self.actor.load_checkpoint(filename)
        self.target_actor.load_checkpoint(filename)
        self.critic.load_checkpoint(filename)
        self.target_critic.load_checkpoint(filename)



