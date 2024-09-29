import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter


# 引用上级目录
import sys
sys.path.append("..")
import grid_env

from model import *

import matplotlib.pyplot as plt

class QNET(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(QNET, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


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


class DPolicyNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(DPolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class ValueNet(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)

class DQN():
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

    def obtain_episode(self, policy, start_state, start_action, length):
        """
        :param policy: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :param length: 一个episode 长度
        :return: 一个列表，其中是字典格式: state,action,reward,next_state,next_action
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)  # 一步动作
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})  #向列表中添加一个字典
        return episode  #返回列表，其中的元素为字典


    def get_data_iter(self, episode, batch_size=64, is_train=True):
        """构造一个PyTorch数据迭代器"""
        reward = []
        state_action = []
        next_state = []
        for i in range(len(episode)):
            reward.append(episode[i]['reward'])
            action = episode[i]['action']
            y, x = self.env.state2pos(episode[i]['state'])
            state_action.append((y, x, action))
            y, x = self.env.state2pos(episode[i]['next_state'])
            next_state.append((y, x))
        reward = torch.tensor(reward).reshape(-1, 1)
        state_action = torch.tensor(state_action)
        next_state = torch.tensor(next_state)
        data_arrays = (state_action, reward, next_state)
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)


    def dqn(self, learning_rate=0.0015, episode_length=5000, epochs=600, batch_size=100, update_step=10):
        policy = self.policy.copy()
        state_value = self.state_value.copy()
        # Initialization: A main network and a target network with the same initial parameter.
        q_net = QNET() #main network
        q_target_net = QNET()
        q_target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.SGD(q_net.parameters(),
                                    lr=learning_rate)
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=episode_length)
        # replay buffer: date_iter
        date_iter = self.get_data_iter(episode, batch_size)
        loss = torch.nn.MSELoss()
        approximation_q_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        i = 0
        rmse_list=[]
        loss_list=[]
        for epoch in range(epochs):
            for state_action, reward, next_state in date_iter:
                i += 1
                q_value = q_net(state_action)
                q_value_target = torch.empty((batch_size, 0))  # 定义空的张量
                for action in range(self.action_space_size):
                    s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
                    q_value_target = torch.cat((q_value_target, q_target_net(s_a)), dim=1)
                q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]
                y_target_value = reward + self.gama * q_star
                l = loss(q_value, y_target_value)
                optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
                l.backward()  # 反向传播更新参数
                optimizer.step()
                if i % update_step == 0 and i != 0:
                    q_target_net.load_state_dict(
                        q_net.state_dict())  # 更新目标网络
                    # policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            loss_list.append(float(l))
            print("loss:{},epoch:{}".format(l, epoch))
            self.policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            self.state_value = np.zeros(shape=self.state_space_size)

            for s in range(self.state_space_size):
                y, x = self.env.state2pos(s)
                for a in range(self.action_space_size):
                    approximation_q_value[s, a] = float(q_net(torch.tensor((y, x, a)).reshape(-1, 3)))
                q_star_index = approximation_q_value[s].argmax()
                self.policy[s, q_star_index] = 1
                self.state_value[s] = approximation_q_value[s, q_star_index]
            rmse_list.append(np.sqrt(np.mean((state_value - self.state_value) ** 2)))
            # policy_rmse = np.sqrt(np.mean((policy - self.policy) ** 2))
        fig_rmse = plt.figure(figsize=(8, 12))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(211)

        # 绘制 rmse 图像
        ax_rmse.plot(rmse_list)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        self.writer.close()
        ax_loss = fig_rmse.add_subplot(212)

        ax_loss.plot(loss_list)
        ax_loss.set_title('loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        plt.show()




if __name__ == '__main__':
    # dqn = PolicyNet()
    # input = torch.tensor([[2, 1], [3, 1]])
    # print(dqn)
    # print(dqn(input))
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    solver = DQN(alpha=0.1, env=gird_world)
    # solver.sarsa()
    # print("env.policy[0, :]:",solver.policy[0, :])
    # for _ in range(20):
    #     a0 = np.random.choice(5, p=solver.policy[0, :] )
    #
    #     print("a0:",a0)

    start_time = time.time()

    solver.dqn()
    print("solver.state_value:", solver.state_value)

    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:", cost_time)

    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()