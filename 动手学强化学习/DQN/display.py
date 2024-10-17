import gym
import torch
import numpy as np
from DQN import DQN
from time import sleep


def dis_to_con(discrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return np.array([discrete_action / (action_dim - 1) * (action_upbound - action_lowbound) + action_lowbound])


lr = 2e-3
hidden_dim = 128
gamma = 0.98
epsilon = 0.0
target_update = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make('Pendulum-v1')

state_dim = env.observation_space.shape[0]
action_dim = 11
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
state_dict = torch.load('dqn_pendulumv1.pth')
agent.q_net.load_state_dict(state_dict)
agent.target_q_net.load_state_dict(state_dict)

state = env.reset()
done = False
agent_return = 0
while not done:
    action = agent.take_action(state)
    action = dis_to_con(action, env, action_dim)
    next_state, reward, done, _ = env.step(action)
    agent_return += reward
    env.render()
    state = next_state
    sleep(0.01)

print('DQN return:', agent_return)


agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, 'DoubleDQN')
state_dict = torch.load('double_dqn_pendulumv1.pth')
agent.q_net.load_state_dict(state_dict)
agent.target_q_net.load_state_dict(state_dict)

state = env.reset()
done = False
agent_return = 0
while not done:
    action = agent.take_action(state)
    action = dis_to_con(action, env, action_dim)
    next_state, reward, done, _ = env.step(action)
    agent_return += reward
    env.render()
    state = next_state
    sleep(0.01)

print('Double DQN return:', agent_return)


agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, 'DuelingDQN')
state_dict = torch.load('dueling_dqn_pendulumv1.pth')
agent.q_net.load_state_dict(state_dict)
agent.target_q_net.load_state_dict(state_dict)

state = env.reset()
done = False
agent_return = 0
while not done:
    action = agent.take_action(state)
    action = dis_to_con(action, env, action_dim)
    next_state, reward, done, _ = env.step(action)
    agent_return += reward
    env.render()
    state = next_state
    sleep(0.01)

print('Dueling DQN return:', agent_return)