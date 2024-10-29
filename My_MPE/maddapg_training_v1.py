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
from maddpg_agent_v1 import Agent

#  Set deviece
print("torch.cuda.is_available:", torch.cuda.is_available())
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))



def multi_observations2state(multi_obs):
    """
    把智能体的观测信息：由字典格式，提取出信息拼接成数组格式。
    """
    state = np.array([])
    for agent_obs in multi_obs.values():  # 访问字典中的值
        state = np.concatenate([state, agent_obs])
    return state
#-----------------------------------------------------------------------------------
NUM_EPISODE = 10000
NUM_STEP = 100
MEMORY_SIZE = 100000
BATCH_SIZE = 64
TARGET_UPDATE_INTERVAL = 200  # 最好是整除NUM_STEP。
# 学习超参数。
GAMMA = 0.99
TAU = 0.01
LR_ACTOR = 1e-2
LR_CRITIC = 1e-2
HIDDEN_DIM = 64


# 1 Initialize the agents
env = simple_tag_v3.parallel_env(render_mode="None", num_good = 1, num_adversaries = 3, num_obstacles = 0, max_cycles = NUM_STEP, continuous_actions = True)
multi_observations, infos = env.reset() # 返回多个agent的初始状态，每个agent的初始状态是一个字典，字典的key是agent的名字，value是agent的初始状态

NUM_AGENT = env.num_agents
agent_name_list = env.agents  # agent_name_list: ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
print("agent_name_list:",agent_name_list)

scenario = "simple_tag_v3"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/models/" + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")

#  1.1 Get obs_dim
obs_dim = []
for agent_obs in multi_observations.values():  # 从多个智能体的观测中一个一个取出，进行处理
    obs_dim.append(agent_obs.shape[0])  # 12， 12， 12，10
state_dim = sum(obs_dim)  #  12 + 12 + 12 + 10 = 46
#  1.2 Get action_dim
action_dim = []
for agent_name in agent_name_list:
    action_dim.append(env.action_space(agent_name).sample().shape[0])  #获取到其中一个agent的action_space


agents = []
for agent_i in range(NUM_AGENT):
    print(f"Initializing agent {agent_i}...")
    print(f"obs_dim{agent_i}:{obs_dim[agent_i]}")
    '''
    obs_dim0:12 ;obs_dim1:12;obs_dim2:12;obs_dim0:12;   问题出现在：！！！obs_dim3:10
    '''
    print(f"初始化agent的obs_dim[agent_i]:{obs_dim[agent_i]}")
    agent = Agent(memory_size = MEMORY_SIZE, obs_dim = obs_dim[agent_i], state_dim = state_dim, n_agent = NUM_AGENT,
                  action_dim = action_dim[agent_i],alpha = LR_ACTOR, beta = LR_CRITIC, fc1_dim = HIDDEN_DIM,
                  fc2_dim = HIDDEN_DIM, gamma = GAMMA, tau = TAU, batch_size = BATCH_SIZE )
    agents.append(agent)

# 2 Main training loop
for episode_i in range(NUM_EPISODE):
    # initailize a random process N for action exploration

    # Receive initial state x
    multi_observations, infos = env.reset()
    episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}  # 每局开始，让每个智能体的完成标志清零。
    for step_i in range(NUM_STEP):
        total_step = episode_i*NUM_STEP + step_i
        # 2.1 Collect action from all agents
        multi_actions = {}
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_observations[agent_name]
            single_action = agent.get_action(single_obs)  #  通过每个智能体自己的观测，分别得到每个agent的action
            multi_actions[agent_name] = single_action

        # 2.2 Execute actions
        multi_next_observations, multi_rewards, multi_done, multi_truncations, infos = env.step(multi_actions)
        state = multi_observations2state(multi_observations)  # state.shape = 46
        # print(f"state.shape:{state.shape}")
        next_state = multi_observations2state(multi_next_observations)
        #  如果跑够步数，那么把done标志位 置1
        if step_i >= NUM_STEP -1:
            multi_done = {agent_name: True for agent_name in agent_name_list}
        # 2.3 Store memory
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_observations[agent_name]
            # print(f"total_step:{total_step}")
            # print(f" {agent_name},multi_next_observations{multi_next_observations[agent_name]}")
            single_next_obs = multi_next_observations[agent_name]
            single_action = multi_actions[agent_name]
            single_reward = multi_rewards[agent_name]
            single_done = multi_done[agent_name]
            agent.replay_buffer.add_memory(single_obs, single_next_obs, state, next_state, \
                                           single_action, single_reward, single_done)
        # 2.4 update brain every fixed steps
        multibatch_obses = []
        multibatch_next_obses = []
        multibatch_states = []
        multibatch_next_states = []
        multibatch_actions = []
        multibatch_next_actions = []
        multibatch_online_actions = []
        multibatch_rewards = []
        multibatch_dones = []
        # 2.4.1 sample a batch of memories
        current_memory_size = min(MEMORY_SIZE, total_step + 1)
        if current_memory_size < BATCH_SIZE:
            batch_idx = range(0, current_memory_size)
        else:
            #  generate a uniform random sample from np.arange(current_memory_size) of size (BATCH_SIZE)
            batch_idx = np.random.choice(current_memory_size, BATCH_SIZE)
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]

            batch_obses, batch_next_obses, batch_states, batch_next_states, \
                batch_actions, batch_rewards, batch_dones = agent.replay_buffer.sample(batch_idx)

            batch_obses_tensor = torch.tensor(batch_obses, dtype=torch.float).to(device)
            batch_next_obses_tensor = torch.tensor(batch_next_obses, dtype=torch.float).to(device)
            batch_states_tensor = torch.tensor(batch_states, dtype=torch.float).to(device)
            batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float).to(device)
            batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.float).to(device)
            batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float).to(device)
            batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float).to(device)

            # multiple + batch
            multibatch_obses.append(batch_obses_tensor)
            multibatch_next_obses.append(batch_next_obses_tensor)
            multibatch_states.append(batch_states_tensor)
            multibatch_next_states.append(batch_next_states_tensor)
            multibatch_actions.append(batch_actions_tensor)

            single_batch_next_actions = agent.target_actor.forward(batch_next_obses_tensor)  #  与up 不同 39：45
            multibatch_next_actions.append(single_batch_next_actions)

            single_batch_online_actions = agent.actor.forward(batch_obses_tensor) #  与up 不同 39：45
            multibatch_online_actions.append(single_batch_online_actions)

            multibatch_rewards.append(batch_rewards_tensor)
            multibatch_dones.append(batch_dones_tensor)

        multibatch_actions_tensor = torch.cat(multibatch_actions, dim=1).to(device)
        multibatch_next_actions_tensor = torch.cat(multibatch_next_actions, dim=1).to(device)
        multibatch_online_actions_tensor = torch.cat(multibatch_online_actions, dim=1).to(device)




        # Update critic and actor Network
        if (total_step+1) % TARGET_UPDATE_INTERVAL == 0:
            for agent_i in range(NUM_AGENT):
                agent = agents[agent_i]

                batch_obses_tensor = multibatch_obses[agent_i]
                batch_next_obses_tensor = multibatch_next_obses[agent_i]
                batch_states_tensor = multibatch_states[agent_i]
                batch_next_states_tensor = multibatch_next_states[agent_i]
                batch_actions_tensor = multibatch_actions[agent_i]
                batch_rewards_tensor = multibatch_rewards[agent_i]
                batch_dones_tensor = multibatch_dones[agent_i]

                # print(f"batch_obses_tensor size: {batch_obses_tensor.size()}")
                # print(f"batch_next_obses_tensor size: {batch_next_obses_tensor.size()}")
                # print(f"batch_states_tensor size: {batch_states_tensor.size()}")
                # print(f"batch_next_states_tensor size: {batch_next_states_tensor.size()}")
                # print(f"batch_actions_tensor size: {batch_actions_tensor.size()}")

                # target critic Network
                print(f"batch_next_obses_tensor.size: {batch_next_states_tensor.size()}")
                print(f"multibatch_next_actions_tensor.size: {multibatch_next_actions_tensor.size()}")

                critic_target_q = agent.target_critic.forward(batch_next_states_tensor,
                                                              multibatch_next_actions_tensor.detach())
                y = (batch_rewards_tensor + (1 - batch_dones_tensor) * agent.gamma * critic_target_q).flatten()
                    # critic Network
                critic_q = agent.critic.forward(batch_states_tensor, multibatch_actions_tensor.detach()).flatten()

                #update critic
                critic_loss = nn.MSELoss(y, critic_q)
                agent.critic.optimizer.zero_grad()
                critic_loss.backward()
                agent.critic.optimizer.step()


                #Update actor Network
                actor_loss = agent.critic.forward(batch_states_tensor,
                                                  multibatch_online_actions_tensor.detach()).flatten() # 这个地方可能会出现问题
                actor_loss = -torch.mean(actor_loss)
                agent.actor.optimizer.zero_grad()
                actor_loss.backward()
                agent.actor.optimizer.step()

                # Update target critic
                for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

                # Update target actor
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

        multi_observations = multi_next_observations
        episode_reward += sum(single_reward for single_reward in multi_rewards.values())
        print(f"total step:{total_step} | episode_reward: {episode_reward}")

    # 3 Render the env
    if (episode_i + 1) % 10 == 0:
        env = simple_tag_v3.parallel_env(render_mode="human",
                                         num_good = 1,
                                         num_adversaries = 3,
                                         num_obstacles = 0,
                                         max_cycles = NUM_STEP,
                                         continuous_actions = True)
        for test_epi_i in range(10):
            multi_observations, infos = env.reset()
            for step_i in range(NUM_STEP):
                multi_actions = {}
                for agent_i, agent_name in enumerate(agent_name_list):
                    agent = agents[agent_i]
                    single_obs = multi_observations[agent_name]
                    single_obs = torch.tensor(data=single_obs, dtype=torch.float).unsqueeze(0).to(device)
                    # single_action = agent.get_action(single_obs)  # 通过每个智能体自己的观测，分别得到每个agent的action
                    multi_actions[agent_name] = single_action
                    multi_next_observations, multi_rewards, multi_done, multi_truncations, infos = env.step(multi_actions)
                    multi_observations = multi_next_observations

    # 4 Save agents' model
    if episode_i == 0:
        hightest_reward = episode_reward
    if episode_reward > hightest_reward:
        print(f"Highest reward updated at episode {episode_i}:{round(episode_reward, 2)}")
        hightest_reward = episode_reward
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            torch.save(agent.actor.state_dict(), f"{agent_path}" + f"agent_{agent_i}_actor_{scenario}_{timestamp}" )

# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     # print("actions:",actions) #  action: array.size = 5
#     multi_next_observations, rewards, terminations, truncations, infos = env.step(actions)
#     # print("observations:",observations)
#     # print("rewards:",rewards)
#     # print("terminations:",terminations)
#     # print("truncations:",truncations)
#     # print("infos:",infos)
env.close()