import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
sys.path.append("..")
import grid_env


'''
首先通过policy iteration计算ground truth，再通过 TD linear 方法去拟合，观察效果。
'''
class TD_learning_with_FunctionApproximation():
    def __init__(self,alpha,env = grid_env.GridEnv):
         self.gamma = 0.9  # discount rate
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


    def get_feature_vector(self, fourier: bool, state: int, ord: int) -> np.ndarray:
        """
        get_feature_vector:   Φ(s)
        :param fourier: 是否使用傅里叶特征函数
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果
        """

        if state < 0 or state >= self.state_space_size:
            raise ValueError("Invalid state value")
        x, y = self.env.state2pos(state) + (1, 1)
        feature_vector = []
        if fourier:
            # 归一化到 -1 到 1
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    feature_vector.append(np.cos(np.pi * (i * x_normalized + j * y_normalized)))

        else:
            # 归一化到 0 到 1
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(y_normalized ** (ord - i) * x_normalized ** j)

        return np.array(feature_vector)

    def get_feature_vector_with_action(self, fourier: bool, state: int, action: int, ord: int) -> np.ndarray:
        """
        get_feature_vector_with_action
        :param fourier: 是否使用傅里叶特征函数
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果
        """

        if state < 0 or state >= self.state_space_size or action < 0 or action >= self.action_space_size:
            raise ValueError("Invalid state/action value")
        feature_vector = []
        y, x = self.env.state2pos(state) + (1, 1)

        if fourier:
            # 归一化到 -1 到 1
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            action_normalized = action / self.action_space_size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    for k in range(ord + 1):
                        feature_vector.append(
                            np.cos(np.pi * (i * x_normalized + j * action_normalized + k * y_normalized)))

        else:
            # 归一化到 0 到 1
            state_normalized = (state - (self.state_space_size - 1) * 0.5) / (self.state_space_size - 1)
            action_normalized = (action - (self.action_space_size - 1) * 0.5) / (self.action_space_size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(state_normalized ** (ord - i) * action_normalized ** j)
        return np.array(feature_vector)

    def td_value_approximation(self, learning_rate=0.0005, epochs=100000, fourier=True, ord=5):
        # self.state_value = self.policy_evaluation(self.policy)
        if not isinstance(learning_rate, float) or not isinstance(epochs, int) or not isinstance(
                fourier, bool) or not isinstance(ord, int):
            raise TypeError("Invalid input type")
        if learning_rate <= 0 or epochs <= 0 or ord <= 0:
            raise ValueError("Invalid input value")
        episode_length = epochs
        start_state = np.random.randint(self.state_space_size)
        start_action = np.random.choice(np.arange(self.action_space_size),
                                        p=self.mean_policy[start_state])
        episode = self.obtain_episode(self.mean_policy, start_state, start_action, length=episode_length)
        dim = (ord + 1) ** 2 if fourier else np.arange(ord + 2).sum()
        w = np.random.default_rng().normal(size=dim)
        rmse = []
        value_approximation = np.zeros(self.state_space_size)
        for epoch in range(epochs):
            reward = episode[epoch]['reward']
            state = episode[epoch]['state']
            next_state = episode[epoch]['next_state']
            target = reward + self.gamma * np.dot(self.get_feature_vector(fourier, next_state, ord), w)
            error = target - np.dot(self.get_feature_vector(fourier, state, ord), w)
            gradient = self.get_feature_vector(fourier, state, ord)
            w = w + learning_rate * error * gradient
            for state in range(self.state_space_size):
                value_approximation[state] = np.dot(self.get_feature_vector(fourier, state, ord), w)
            rmse.append(np.sqrt(np.mean((value_approximation - self.state_value) ** 2)))
            print(epoch)
        X, Y = np.meshgrid(np.arange(1, 6), np.arange(1, 6))
        Z = self.state_value.reshape(5, 5)
        Z1 = value_approximation.reshape(5, 5)
        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('State Value')
        z_min = -5
        z_max = -2
        ax.set_zlim(z_min, z_max)
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Value Approximation')
        ax1.set_zlim(z_min, z_max)
        fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(111)

        # 绘制 rmse 图像
        ax_rmse.plot(rmse)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        plt.show()
        return value_approximation

if __name__ == "__main__":
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                               forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                               render_mode='')

    solver = TD_learning_with_FunctionApproximation(alpha =0.1, env = gird_world)
    solver.td_value_approximation()



    solver.show_state_value(state_value=solver.td_value_approximation(), y_offset=-0.25)

    solver.env.render()