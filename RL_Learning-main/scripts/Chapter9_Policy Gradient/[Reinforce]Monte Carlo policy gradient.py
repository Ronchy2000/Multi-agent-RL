import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter
from torch.utils import data
import torch
import torch.nn as nn


# 引用上级目录
import sys
sys.path.append("..")
import grid_env

"""
REINFORCE algorithm

"""

"Define policy NN"
class PolicyNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=5):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class REINFORCE():
    def __init__(self,alpha,env = grid_env.GridEnv):
        self.gama = 0.9  # discount rate
        self.alpha = alpha  #learning rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # [-10,-10,0,1]  reward list
        self.state_value = np.zeros(shape=self.state_space_size)  # 一维列表
        print("self.state_value:", self.state_value)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 二维： state数 x action数
        self.mean_policy = np.ones(     #self.mean_policy shape: (25, 5)
            shape=(self.state_space_size, self.action_space_size)) / self.action_space_size  # 平均策略，即取每个动作的概率均等
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))
        print('----------------------------------------------------------------')

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obtain_episode_net(self, policy_net, start_state, start_action):
        """
        :param policy_net: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :return: 一个列表，其中是字典格式: state,action,reward,next_state,next_action
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        done = False
        while not done:
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)  # 一步动作
            next_state = self.env.pos2state(self.env.agent_location)
            x, y = self.env.state2pos(next_state) / self.env.size
            prb = policy_net(torch.tensor((x, y)).reshape(-1, 2))[0]
            next_action = np.random.choice(np.arange(self.action_space_size), p = prb.detach().numpy())
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})  # 向列表中添加一个字典
        return episode

    def reiniforce(self, epochs=20000):
        policy_net = PolicyNet()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.alpha)
        for epoch in range(epochs):
            prb = policy_net(torch.tensor((0, 0)).reshape(-1, 2))[0]
            print("epoch:{} , prb:{}".format(epoch, prb))
            start_action = np.random.choice(np.arange(self.action_space_size), p=prb.detach().numpy())
            episode = self.obtain_episode_net(policy_net, start_state=0, start_action=start_action)
            # print("eposode:", episode)
            if len(episode) < 10 :
                g = -100
            else:
                g = 0
            optimizer.zero_grad()  # 清零梯度
            for step in reversed(range(len(episode))):
                reward = episode[step]['reward']
                state = episode[step]['state']
                action = episode[step]['action']
                if len(episode) > 1000:
                    print(g, reward)
                g = self.gama * g + reward
                self.qvalue[state, action] = g
                x ,y = self.env.state2pos(state)/self.env.size
                prb = policy_net(torch.tensor((x, y)).reshape(-1, 2))[0]
                log_prob = torch.log(prb[action])
                loss = -log_prob * g
                loss.backward() #反向传播计算梯度
            self.writer.add_scalar("loss", float(loss.detach()), epoch)
            self.writer.add_scalar('g', g, epoch)
            self.writer.add_scalar('episode_length', len(episode), epoch)
            # print(epoch, len(episode), g)
            optimizer.step()
        for s in range(self.state_space_size):
            x, y = self.env.state2pos(s) / self.env.size
            prb = policy_net(torch.tensor((x, y)).reshape(-1, 2))[0]
            self.policy[s,:] = prb.copy()
        self.writer.close()

if __name__ == '__main__':
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    solver = REINFORCE(alpha=0.001, env=gird_world)
    start_time = time.time()

    solver.reiniforce()
    print("solver.state_value:", solver.state_value)


    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:", cost_time)
    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()