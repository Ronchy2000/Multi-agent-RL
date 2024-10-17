import gym
import torch
import random
import numpy as np
from DQN import DQN
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
import os
# 将上级目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import rl_utils
from rl_utils import *

def dis_to_con(discrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return np.array([discrete_action / (action_dim - 1) * (action_upbound - action_lowbound) + action_lowbound])


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0

    for i in range(10):
        with tqdm(total=num_episodes // 10, desc='Iteration %d' % i) as pbar:
            for i_episode in range(num_episodes // 10):
                episode_return = 0
                state, *_ = env.reset()
                done = False
                while not done:
                    # print("state", state)
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)

                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, done, *_ = env.step(action_continuous)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = dict(
                            states=b_s,
                            actions=b_a,
                            rewards=b_r,
                            next_states=b_ns,
                            dones=b_d
                        )
                        agent.update(transition_dict)
                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list



lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# env_name = 'Pendulum-v1'
env_name = "CartPole-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 11

print("DQN")
random.seed(0)
np.random.seed(0)
env.reset(seed = 0)  # 新版gymnausim
# env.seed(0)  #旧版gym
torch.manual_seed(0)

replay_buffer = ReplayBuffer(buffer_size)
print("replay_buffer建立成功！", replay_buffer)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

torch.save(agent.q_net.state_dict(), 'dqn_pendulumv1.pth')
episodes_list = list(range(len(return_list)))
mv_returns = moving_average(return_list, 5)
plt.plot(episodes_list, mv_returns)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns on {}'.format(env_name))
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN Q Value on {}'.format(env_name))
plt.show()

# --------------------------------------------------------
#
# print("Double DQN")
# random.seed(0)
# np.random.seed(0)
# env.seed(0)
# torch.manual_seed(0)
#
# replay_buffer = ReplayBuffer(buffer_size)
# agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, "DoubleDQN")
# return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)
#
# torch.save(agent.q_net.state_dict(), 'double_dqn_pendulumv1.pth')
# episodes_list = list(range(len(return_list)))
# mv_returns = moving_average(return_list, 5)
# plt.plot(episodes_list, mv_returns)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('Double DQN Returns on {}'.format(env_name))
# plt.show()
# --------------------------------------------------------
# frames_list = list(range(len(max_q_value_list)))
# plt.plot(frames_list, max_q_value_list)
# plt.axhline(0, c='orange', ls='--')
# plt.axhline(10, c='red', ls='--')
# plt.xlabel('Frames')
# plt.ylabel('Q value')
# plt.title('Double DQN Q value on {}'.format(env_name))
# plt.show()
#
# print("Dueling DQN")
# random.seed(0)
# np.random.seed(0)
# env.seed(0)
# torch.manual_seed(0)
#
# replay_buffer = ReplayBuffer(buffer_size)
# agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, "DuelingDQN")
# return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)
#
# torch.save(agent.q_net.state_dict(), 'dueling_dqn_pendulumv1.pth')
# episodes_list = list(range(len(return_list)))
# mv_returns = moving_average(return_list, 5)
# plt.plot(episodes_list, mv_returns)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('Dueling DQN Returns on {}'.format(env_name))
# plt.show()
#
# frames_list = list(range(len(max_q_value_list)))
# plt.plot(frames_list, max_q_value_list)
# plt.axhline(0, c='orange', ls='--')
# plt.axhline(10, c='red', ls='--')
# plt.xlabel('Frames')
# plt.ylabel('Q value')
# plt.title('Dueling DQN Q value on {}'.format(env_name))
# plt.show()