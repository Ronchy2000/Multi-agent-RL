import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
sys.path.append("..")
import grid_env

class MC_epsilon_greedy:
    def __init__(self, env = grid_env.GridEnv):
        self.gama = 0.9   #discount rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list  # [-10,-10,0,1]  reward list
        self.state_value = np.zeros(shape=self.state_space_size) #一维列表
        print("self.state_value:",self.state_value)
        #Q表和policy 维数一样
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 二维： state数 x action数
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size  #平均策略，即取每个动作的概率均等
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}" .format(self.action_space_size ,self.state_space_size) )
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,self.qvalue.shape, self.mean_policy.shape))

        print('----------------------------------------------------------------')
    '''
        定义可视化grid world所需的函数
        def show_policy(self)
        def show_state_value(self, state_value, y_offset=0.2):
        def obtain_episode(self, policy, start_state, start_action, length):
    '''
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
        :param length: 一个episode 最大长度
        :return: 一个 state,action,reward,next_state,next_action 列表，其中是字典格式
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),  #[0, len(policy[next_state]) 中随机抽一个随机数
                                           p=policy[next_state])  #p参数的例子： p=[0.1, 0.2, 0.3, 0.1, 0.3]的概率从 [0,1,2,3,4]这四个数中选取3个数
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})  #向列表中添加一个字典
            if done:
                break
        return episode

    def mc_epsilon_greedy(self, episodes, episode_length, epsilon = 0.5 ):
        # 初始化Returns和Num计数器
        returns = np.zeros(self.qvalue.shape)  # 初始化回报累计
        num_visits = np.zeros(self.qvalue.shape, dtype=int)  # 初始化访问次数

        for _ in range(episodes):
            # Episode generation
            start_state = np.random.randint(self.state_space_size)  # 随机选择起始状态
            start_action = np.random.choice(np.arange(self.action_space_size),  # 随机选择起始动作
                                            p=self.policy[start_state])

            episode = self.obtain_episode(self.policy, start_state, start_action,
                                          episode_length)  # 获取一个episode

            # 对于每个step的回报累积和访问次数更新
            for step in reversed(episode):  # 逆序遍历，从T-1到0
                state, action, reward = step["state"], step["action"], step["reward"]
                G = reward  # 当前步的即时奖励
                for rt in episode[::-1][episode.index(step):]:  # 从当前步开始反向累加未来奖励
                    G = self.gama * G + rt["reward"]  # 累积折扣回报
                returns[state, action] += G  # 更新累积回报
                num_visits[state, action] += 1  # 更新状态动作对的访问次数

            # Policy evaluation
            self.qvalue = np.divide(returns, num_visits, where=num_visits != 0)  # 避免除以零错误
            # Policy improvement
            best_actions = np.argmax(self.qvalue, axis=1)  # 找到每个状态下最优的动作
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    # self.policy[state, action] = (1 - epsilon + epsilon / self.action_space_size) * (
                    #             action == best_actions[state]) + \
                    #                              (epsilon / self.action_space_size) * (action != best_actions[state])
                    self.policy[state, :] = 0  # 先将所有动作概率设为0
                    self.policy[state, best_actions[state]] = 1  # 最优动作概率设为1


if __name__ == "__main__":
    episodes = 1000
    episode_length = 2000
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    solver = MC_epsilon_greedy(gird_world)
    start_time = time.time()

    # solver.state_value = solver.mc_exploring_starts_first_visit(length=episode_length)
    solver.mc_epsilon_greedy(episodes, episode_length)  # 修改后，利用tqdm显示epoch进度

    end_time = time.time()
    cost_time = end_time - start_time
    print("episode_length:{} that the cost_time is:{}".format(episode_length, round(cost_time, 2)))

    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    gird_world.plot_title("Episode_length = " + str(episode_length))
    gird_world.render()
    # gird_world.render_clear()
    print("--------------------")
