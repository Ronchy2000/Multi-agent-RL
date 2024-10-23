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

# 1 Initialize the agents
env = simple_tag_v3.parallel_env(render_mode="human", num_good = 1, num_adversaries = 3, num_obstacles = 0, max_cycles = 25, continuous_actions = True)
multi_observations, infos = env.reset() # 返回多个agent的初始状态，每个agent的初始状态是一个字典，字典的key是agent的名字，value是agent的初始状态

NUM_AGENT = env.num_agents
agent_name_list = env.agents

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