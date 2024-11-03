# -*- coding: utf-8 -*-
#
# @Time : 2024-10-30 11:04:26
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : main.py
# @Software: PyCharm
# @Description: None

import random
from collections import namedtuple, deque

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
# from pettingzoo.mpe import simple_adversary_v3
from pettingzoo.mpe import simple_tag_v3
from psutil import virtual_memory

from ReplayBuffer import *
from NN_Module import *
from MADDPG import *


'''
Set up parameters and initialize variables
'''
# Set up Parameters
NUM_EPISODES = 50000
LEN_EPISODES = 25  # The maximum length of each episode
BUFFER_SIZE = 100_000
HIDDEN_DIM = 64  # denotes the dimension of the hidden layers in the actor and critic networks.
ACTOR_LR = 1e-2
CRITIC_LR = 1e-2
GAMMA = 0.95  # determines the discount factor for future rewards.
TAU = 1e-2  # sets the soft update coefficient for target network updates.
BATCH_SIZE = 1024
UPDATE_INTERVAL = 100  # determines the number of steps before performing an update on the networks.
MINIMAL_SIZE = 4000  # is a minimum threshold for the replay buffer size before starting the training.
# Initialize Environment
# env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=False)
env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=25, continuous_actions=True)
obs, infos = env.reset()
'''
test_simple_tag_env:
'''
# while env.agents:
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents} #字典形式，多智能体的联合动作  采样。
#     print("actions:",actions)
#     observation,rewards, dones, truncations, infos = env.step(actions)
#     print("observation",observation)
#     print("rewards",rewards)
#     print("dones",dones)
#     print("truncations",truncations)
#     print("infos",infos)

# env.close()



# env.reset() # 原来的，保留
# Initialize Replay Buffer
buffer = ReplayBuffer(BUFFER_SIZE)  # stores experiences from the environment.
# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Return List
return_list = []
# Initialize Total Step
total_step = 0
# Initialize MADDPG
# 1. Get State and Action Dimensions from the Environment
state_dims = [env.observation_space(env.agents[i]).shape[0] for i in range(env.num_agents)]
print("state_dims:",state_dims)
action_dims = [env.action_space(env.agents[i]).n for i in range(env.num_agents)]
print("action_dims:",action_dims)
critic_dim = sum(state_dims) + sum(action_dims)  # calculates the total dimension of the critic network's input
# 2. Create Center Controller
maddpg = MADDPG(state_dims, action_dims, critic_dim, HIDDEN_DIM, ACTOR_LR, CRITIC_LR, device, GAMMA, TAU)
# maddpg is created as an instance of the MADDPG class using the specified dimensions, learning rates and other parameters.


def evaluate(num_agents, maddpg, num_episode=10, len_episode=25):
    """Evaluate the strategies for learning, and no exploration is undertaken at this time.

    Args:
        num_agents: The number of the agents.
        maddpg: The Center Controller.
        num_episode: The number of episodes.
        len_episode: The length of each episode.

    Returns: Returns list.

    """
    #     env = simple_adversary_v3.parallel_env(N=num_agents, max_cycles=25, continuous_actions=False)

    # pusuit-evador env.
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=25,
                                     continuous_actions=False)
    env.reset()
    returns = np.zeros(env.max_num_agents)
    # Create an array returns of zeros with a length equal to the number of agents in the environment.
    # This array will store the cumulative returns for each agent.
    for episode in range(num_episode):
        states_dict, rewards_dict = env.reset()
        states = [state for state in states_dict.values()]
        for episode_step in range(len_episode):
            actions = maddpg.take_action(states, explore=False)
            # Take actions using the MADDPG agent (maddpg.take_action) based on the current states.
            actions_SN = [np.argmax(onehot) for onehot in actions]
            actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
            next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(actions_dict)
            # Take a step in the environment by passing the actions dictionary to env.step.
            rewards = [reward for reward in rewards_dict.values()]
            next_states = [next_state for next_state in next_states_dict.values()]
            states = next_states
            rewards = np.array(rewards)
            returns += rewards / num_episode
    env.close()
    return returns.tolist()


'''
Auxiliary function
'''
def sample_rearrange(x, device):
    """Rearrange the transition in the sample."""
    rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
    return [torch.FloatTensor(np.vstack(attribute)).to(device) for attribute in rearranged]
    # np.vstack is used to vertically stack the elements in each sublist before converting them to a tensor.
    # The resulting tensors are then converted to the appropriate device (GPU).
#===============================================================================================================
for episode in range(NUM_EPISODES):
    # Reset the Environment
    states_dict, _ = env.reset()
    states = [state for state in states_dict.values()]
    # Start a New Game
    for episode_step in range(LEN_EPISODES):
        # Initial States and Actions
        actions = maddpg.take_action(states, explore=True)
        actions_SN = [np.argmax(onehot) for onehot in actions]
        actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
        # Step
        next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(actions_dict)
        # Add to buffer
        rewards = [reward for reward in rewards_dict.values()]
        next_states = [next_state for next_state in next_states_dict.values()]
        terminations = [termination for termination in terminations_dict.values()]
        buffer.add(states, actions, rewards, next_states, terminations)
        # Update States
        states = next_states
        # Count += 1
        total_step += 1
        # When the replay buffer reaches a certain size and the number of steps reaches the specified update interval
        # 1. Sample from the replay buffer.
        # 2. Update Actors and Critics.
        # 3. Update Target networks' parameters.
        if len(buffer) >= MINIMAL_SIZE and total_step % UPDATE_INTERVAL == 0:
            sample = buffer.sample(BATCH_SIZE)
            sample = [sample_rearrange(x, device) for x in sample]
            # Update Actors and Critics
            for agent_idx in range(env.max_num_agents):
                maddpg.update(sample, agent_idx)
            # Update Target Parameters
            maddpg.update_all_targets_params()
    # After every 100 rounds, use the evaluate function to evaluate the trained agents, get the reward list ep_returns, and add it to return_list.
    if (episode + 1) % 100 == 0:
        episodes_returns = evaluate(env.max_num_agents - 1, maddpg, num_episode=100)
        return_list.append(episodes_returns)
        print(f"Episode: {episode + 1}, {episodes_returns}")
# Close the Environment
env.close()
return_array = np.array(return_list)


#====================================================================
'''
Plot
'''
num_agents = return_array.shape[1]
df = pd.DataFrame(return_array, columns=[f'agent_{i - 1}' if i > 0 else 'adversary_0' for i in range(num_agents)])
df.to_excel('return_data.xlsx', index=False)

plt.figure(figsize=(10, 5))
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.title('Return vs Episodes for each Agent')
plt.grid(True)

for agent in range(num_agents):
    agent_returns = return_array[:, agent]
    if agent == 0:
        plt.plot(range(100, len(agent_returns) * 100 + 100, 100), return_array[:, 0], label=f'adversary_0')
    else:
        plt.plot(range(100, len(agent_returns) * 100 + 100, 100), agent_returns, label=f'agent_{agent - 1}')

plt.legend()
plt.savefig('return_plot.png')
plt.show()