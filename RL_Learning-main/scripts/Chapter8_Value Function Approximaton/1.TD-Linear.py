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
有点问题，没有解决2024.8.30
'''
class TD_learning_with_FunctionApproximation():
    def __init__(self,alpha,env = grid_env.GridEnv):
         self.gamma = 0.9  # discount rate
         self.learning_rate = alpha  #learning rate
         self.env = env
         self.action_space_size = env.action_space_size
         self.state_space_size = env.size ** 2
         self.reward_space_size, self.reward_list = len(
             self.env.reward_list), [-1,-1,0,1]  # [-10,-10,0,1]  reward list
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
        :param ord: 特征函数最高阶次数/傅里叶(对应书)
        :return: 多项式特征向量
        """
        if state < 0 or state >= self.state_space_size:
            raise ValueError("Invalid state value")

        x, y = self.env.state2pos(state) + (1, 1)
        feature_vector = []


        if fourier:
            # 傅里叶feature vector
            # 归一化到 [-1 ,1]
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    feature_vector.append(np.cos(np.pi * (i * x_normalized + j * y_normalized)))

        else:
            #多项式 featrue vector
            # 归一化到 [0,1] ;
            # 将数据中心化到 [0, self.env.size - 1] 区间的中心位置,确保数据在归一化后不会偏向区间的一端。
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            # 初始化特征向量
            feature_vector = [1]  # 特征向量的第一个元素总是常数1
            for i in range(1, ord + 1):  # 从1到ord阶
                for j in range(i + 1):  # j表示y的指数，i-j表示x的指数
                    feature_vector.append((x_normalized ** (i - j)) * (y_normalized ** j))  #[1, x, y, x^2, xy, y^2]
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


    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
        """
        迭代求解贝尔曼公式 得到 state value tolerance 和 steps 满足其一即可
        :param policy: 需要求解的policy
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止计算 此时若是policy iteration 则算法变为 truncated iteration
        :return: 求解之后的收敛值
        """
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > tolerance:
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qvalue(state_value=state_value_k.copy(),
                                                                           state=state,
                                                                           action=action)  # bootstrapping
                state_value_k[state] = value
        return state_value_k
    def calculate_qvalue(self, state, action, state_value):
        """
        计算qvalue elementwise形式
        :param state: 对应的state
        :param action: 对应的action
        :param state_value: 状态值
        :return: 计算出的结果
        """
        qvalue = 0
        for i in range(self.reward_space_size):
            qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]
        for next_state in range(self.state_space_size):
            qvalue += self.gamma * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue

    def td_state_value_hat(self, epochs=5000, fourier=False, ord=1):  # False 时为多项式feature vector
        self.state_value = self.policy_evaluation(self.policy)
        if not isinstance(self.learning_rate, float) or not isinstance(epochs, int) or not isinstance(
                fourier, bool) or not isinstance(ord, int):
            raise TypeError("Invalid input type")
        if self.learning_rate <= 0 or epochs <= 0 or ord <= 0:
            raise ValueError("Invalid input value")

        dim = (ord + 1) ** 2 if fourier else np.arange(ord + 2).sum()  #条件表达式;  计算特征向量的维度
        w = np.random.default_rng().normal(size=dim) # 初始化权重参数; 生成了一个长度为 dim 的向量，其中每个元素都服从标准正态分布（均值为 0，方差为 1）
        print("feature vector w:",w) # parameter

        rmse = []  #均方根误差RMSE（Root Mean Square Error） RMSE = √(Σ(yi - Ŷi)²/n)
        value_hat = np.zeros(self.state_space_size)

        for epoch in range(epochs):
            start_state = np.random.randint(self.state_space_size)
            start_action = np.random.choice(np.arange(self.action_space_size),
                                            p=self.mean_policy[start_state])
            episode = self.obtain_episode(self.mean_policy, start_state, start_action, length=epochs)
            for sample in episode:
                reward = sample['reward']
                state = sample['state']
                next_state = sample['next_state']
                # target = reward + self.gamma * np.dot(self.get_feature_vector(fourier, next_state, ord), w) # 括号内第一项，第二项求点积,即 phi(s_t+1)*w
                # error = target - np.dot(self.get_feature_vector(fourier, state, ord), w)  #括号内第一项，第二项求点积即 phi(s_t)*w
                # gradient = self.get_feature_vector(fourier, state, ord)  # phi(s)*w的梯度为phi(s)，即feature vector本身
                # w = w + learning_rate * error * gradient
                #书中的TD-Linear公式
                w += (self.learning_rate*
                      (reward
                    + self.gamma*np.dot(self.get_feature_vector(fourier, next_state, ord),w)
                    - np.dot(self.get_feature_vector(fourier, state, ord),w) ))

            for state in range(self.state_space_size):
                value_hat[state] = np.dot(self.get_feature_vector(fourier, state, ord), w)
            rmse.append(np.sqrt(np.mean((value_hat - self.state_value) ** 2)))
            print(epoch)

        X, Y = np.meshgrid(np.arange(1, 6), np.arange(1, 6))  # position on grid world.
        Z = self.state_value.reshape(5, 5)
        Z1 = value_hat.reshape(5, 5)
        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax = fig.add_subplot(121, projection='3d')

        ax.plot_surface(X, Y, Z)
        ax.set_title('True State Value')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('State Value')
        z_min,z_max = -5,-2
        ax.set_zlim(z_min, z_max)
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z1)
        ax1.set_title('Estimated State Value')
        ax1.set_xlabel('X')
        ax1.set_zlim(z_min, z_max)
        fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(111)

        # 绘制 rmse 图像
        ax_rmse.plot(rmse)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        plt.show()
        return value_hat

if __name__ == "__main__":
    print("Creating grid world")
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')

    print("Creating solver")
    solver = TD_learning_with_FunctionApproximation(alpha=0.0005, env=gird_world)  # 实例化5

    print("Calculating state value hat")
    state_value_hat = solver.td_state_value_hat()
    solver.show_state_value(state_value=state_value_hat, y_offset=-0.25)
    solver.show_state_value(state_value=solver.state_value, y_offset=-0.25)
    print("state_value_hat:", state_value_hat)
    print("solver.state_value:", solver.state_value)
    gird_world.render()