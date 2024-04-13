import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

import grid_env

class  class_policy_iteration:
    def __init__(self,env: grid_env.GridEnv):
        self.gama = 0.9  # discount rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2  # 幂运算，grid world的尺寸 如 5 ** 2 = 25的网格世界。
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # 父类中：self.reward_list = [0, 1, -10, -10]
        # state_value
        self.state_value = np.zeros(shape=self.state_space_size)  # 1维数组
        # action value -> Q-table
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 25 x 5

        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))
        print("\n分别是non-forbidden area, target area, forbidden area 以及撞墙:")
        print("self.reward_space_size:{},self.reward_list:{}".format(self.reward_space_size, self.reward_list))
        print('----------------------------------------------------------------')

    def policy_iteration_new(self,tolerance = 0.001,steps=100):
        pass


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
        :param length: episode 长度
        :return: 一个 state,action,reward,next_state,next_action 序列
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
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode


if __name__ == "__main__":
    print("-----Begin!-----")
    gird_world2x2 = grid_env.GridEnv(size=5, target=[1, 1],
                           forbidden=[[1, 0]],
                           render_mode='')
    solver = class_policy_iteration(gird_world2x2)
    start_time = time.time()