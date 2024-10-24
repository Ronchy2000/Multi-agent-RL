# -*- coding: utf-8 -*-
# @Time : 2024-10-23 20:40:27
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : maddapg_training_v1.py
# @Software: PyCharm
# @Description: None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3

#  Set deviece
print("torch.cuda.is_available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device) )



def multi_observations2state(multi_obs):
    """
    把智能体的观测信息：由字典格式，提取出信息拼接成数组格式。
    """
    state = np.array([])
    for agent_obs in multi_obs.values():  # 访问字典中的值
        state = np.concatenate([state,agent_obs])
    return state
#-----------------------------------------------------------------------------------
NUM_EPISODE = 10000
NUM_STEP = 200

# 1 Initialize the agents
env = simple_tag_v3.parallel_env(render_mode="human", num_good = 1, num_adversaries = 3, num_obstacles = 0, max_cycles = 25, continuous_actions = True)
multi_observations, infos = env.reset() # 返回多个agent的初始状态，每个agent的初始状态是一个字典，字典的key是agent的名字，value是agent的初始状态

NUM_AGENT = env.num_agents
agent_name_list = env.agents  # agent_name_list: ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
print("agent_name_list:",agent_name_list)

#  1.1 Get obs_dim
obs_dim = []
for agent_obs in multi_observations.vaules():  # 从多个智能体的观测中一个一个取出，进行处理
    obs_dim.append(agent_obs.shape[0])
state_dim = sum(obs_dim)
#  1.2 Get action_dim
action_dim = []
for agent_name in agent_name_list:
    action_dim.append(env.action_space(agent_name).sample().shape[0])  #获取到其中一个agent的action_space


agents = []
for agent_i in range(NUM_AGENT):
    agent = Agent(obs_dim = obs_dim, action_dim = action_dim)  # TODO
    agents.append(agent)

# 2 Main training loop
for episode_i in range(NUM_EPISODE):
    # initailize a random process N for action exploration

    # Receive initial state x
    multi_observations, infos = env.reset()
    epiode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}  # 每局开始，让每个智能体的完成标志清零。
    for step_i in range(NUM_STEP):

        # 2.1 Collect action from all agents
        multi_actions = {}
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_observations[agent_name]
            single_action = agent.get_action(single_obs)  #  通过每个智能体自己的观测，分别得到每个agent的action
            multi_actions[agent_name] = single_action

        # 2.2 Execute actions
        multi_next_observations, multi_rewards, multi_done, multi_truncations, infos = env.step(multi_actions)
        next_state = multi_observations2state(multi_observations)
        #  如果跑够步数，那么把done标志位 置1
        if step_i >= NUM_STEP -1:
            multi_done = {agent_name: True for agent_name in agent_name_list}
        # 2.3 Store memory
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_observations[agent_name]
            single_next_obs = multi_next_observations[agent_name]
            single_action = multi_actions[agent_name]
            single_reward = multi_rewards[agent_name]
            single_done = multi_done[agent_name]
            agent_replay_buffer.add_memory(single_obs, single_action, single_reward, \
                                           single_next_obs, single_done)
        # update brain every fixed steps


        multi_observations = multi_next_observations

# 3 Render the env

# 4 Save agents' model


while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # print("actions:",actions) #  action: array.size = 5
    multi_next_observations, rewards, terminations, truncations, infos = env.step(actions)
    # print("observations:",observations)
    # print("rewards:",rewards)
    # print("terminations:",terminations)
    # print("truncations:",truncations)
    # print("infos:",infos)
env.close()