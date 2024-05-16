import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
sys.path.append("..")
import grid_env


'''
MC Basic 是个Model free 的方法，与value iteration和 Policy iteration对比，数据是MC的必需品。


'''
class MC_Exploring_Starts:
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
        :param length: 一个episode 长度
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
        return episode

    def mc_exploring_starts_simple(self, length=50, epochs=10):
        """
        :param length: 每一个 state-action 对的长度
        :return:
        """
        for epoch in range(epochs):
            episode = self.obtain_episode(self.policy, state, action, length)  # policy is mean policy

            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(self.policy, state, action, length)  # policy is mean policy
                    print("obtain_episode,type:,{}; {}".format(type(episode[0]), episode))
                    # Policy evaluation:
                    sum_qvalue = 0
                    for i in range(len(episode) - 1):
                        sum_qvalue += episode[i]['reward']
                    self.qvalue[state][action] = sum_qvalue

                # Policy improvement:
                max_index = np.argmax(self.qvalue[state]) # qvalue_star
                max_qvalue = np.max(self.qvalue[state]) #action_star


    def mc_exploring_starts(self, length=10):
        time_start = time.time()
        policy = self.mean_policy.copy()
        qvalue = self.qvalue.copy()
        returns = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        while np.linalg.norm(policy - self.policy, ord=1) > 0.001:
            policy = self.policy.copy()
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    visit_list = []
                    g = 0
                    episode = self.obtain_episode(policy=self.policy, start_state=state, start_action=action,
                                                  length=length)
                    for step in range(len(episode) - 1, -1, -1):
                        reward = episode[step]['reward']
                        state = episode[step]['state']
                        action = episode[step]['action']
                        g = self.gama * g + reward
                        ##Exploring Starts
                        # first visit
                        if [state, action] not in visit_list:
                            visit_list.append([state, action])
                            returns[state][action].append(g)
                            qvalue[state, action] = np.array(returns[state][action]).mean()
                            qvalue_star = qvalue[state].max()
                            action_star = qvalue[state].tolist().index(qvalue_star)
                            self.policy[state] = np.zeros(shape=self.action_space_size).copy()
                            self.policy[state, action_star] = 1
            print(np.linalg.norm(policy - self.policy, ord=1))

        time_end = time.time()
        print("mc_exploring_starts cost time:" + str(time_end - time_start))
