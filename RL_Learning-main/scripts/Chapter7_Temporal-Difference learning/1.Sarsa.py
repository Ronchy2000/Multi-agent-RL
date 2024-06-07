import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
sys.path.append("..")
import grid_env

"""
SARSA: State - action - reward - state - action

TD learning of acton values: Sarsa  ->  directly estimate action values.
"""
class Sarsa():
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

    '''
        Learn an optimal policy that can lead the agent to the target state from an initial state s0.
    '''
    def Sarsa_alg(self,initial_location, epsilon = 0.1):
        total_rewards = []
        episode_lengths = []
        initial_state = self.env.pos2state(initial_location)
        print("initial_state:", initial_state)
        for episode_num in range(1000): # episode_num
            self.env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            print("episode_num:",episode_num)

            state = initial_state
            action = np.random.choice(a=np.arange(self.action_space_size),
                                      p=self.policy[state, :])  # Generate a0 at s0 following π0(s0)
            while not done:  #If s_t is not the target state, do
                episode_length += 1
                _, reward, done, _, _ = self.env.step(action) #Collect an experience sample (rt+1, st+1, at+1)
                #S
                next_state = self.env.pos2state(self.env.agent_location)
                # print("next_state:",next_state, "self.env.agent_location:",self.env.agent_location)
                #A
                next_action = np.random.choice(np.arange(self.action_space_size),
                                               p=self.policy[next_state,:])
                total_reward += reward
                #Update q-value for (st, at):
                self.qvalue[state][action] = self.qvalue[state, action] - self.alpha * (self.qvalue[state, action] - (reward + self.gama * self.qvalue[next_state, next_action]) )
                #update policy
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - epsilon + (epsilon / self.action_space_size)

                    else:
                        self.policy[state, a] = epsilon / self.action_space_size
                action = next_action
                state = next_state
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        return total_rewards,episode_lengths

if __name__ =="__main__":
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    solver = Sarsa(alpha =0.1, env = gird_world)
    # solver.sarsa()
    # print("env.policy[0, :]:",solver.policy[0, :])
    # for _ in range(20):
    #     a0 = np.random.choice(5, p=solver.policy[0, :] )
    #
    #     print("a0:",a0)

    start_time = time.time()

    initial_location = [0,0]
    total_rewards, episode_lengths = solver.Sarsa_alg(initial_location = initial_location)


    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:",cost_time)
    print(len(gird_world.render_.trajectory))

    initial_state = solver.env.pos2state(initial_location)
    print("训练后的policy结果为:\n",solver.policy[initial_state,:])
    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    # gird_world.plot_title("Episode_length = " + str(i))
    gird_world.render()
    # gird_world.render_clear()
    print("--------------------")
    print("Plot")
    # 绘制第一个图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(total_rewards) + 1), total_rewards,   # 空心，设置填充色为透明
             markeredgecolor='blue',  # 边框颜色为蓝色
             markersize=10,
             linestyle='-', color='blue',label = "total_rewards")
    plt.xlabel('Episode index', fontsize=12)
    plt.ylabel('total_rewards', fontsize=12)

    # 绘制第二个图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_lengths) + 1), episode_lengths,  # 空心，设置填充色为透明
             markeredgecolor='blue',  # 边框颜色为蓝色
             markersize=10,
             linestyle='-', color='blue',label = "episode_length")
    plt.xlabel('Episode index', fontsize=12)
    plt.ylabel('episode_length', fontsize=12)

    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()
