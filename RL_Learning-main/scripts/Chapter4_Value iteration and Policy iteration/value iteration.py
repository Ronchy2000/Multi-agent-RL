import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录 - 使用绝对路径，跨平台兼容
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grid_env


class class_value_iteration():
    def __init__(self, env: grid_env.GridEnv):
        self.gama = 0.9   #discount rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2  #幂运算，grid world的尺寸 如 5 ** 2 = 25的网格世界。
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list  #父类中：self.reward_list = [0, 1, -10, -10]
        #state_value
        self.state_value = np.zeros(shape=self.state_space_size)  # 1维数组
        #action value -> Q-table
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size)) # 25 x 5

        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("../logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}" .format(self.action_space_size ,self.state_space_size) )
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,self.qvalue.shape, self.mean_policy.shape))
        print("\n分别是non-forbidden area, target area, forbidden area 以及撞墙:")
        print("self.reward_space_size:{},self.reward_list:{}".format(self.reward_space_size,self.reward_list))
        print('----------------------------------------------------------------')

    def value_iteration_new(self, tolerance=0.001, steps=100):
        """
        迭代求解最优贝尔曼公式 得到 最优state value tolerance 和 steps 满足其一即可
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止 建议将此变量设置大一些
        :return: 剩余迭代次数
        """
        # 初始化 V0 为 1
        state_value_k = np.ones(self.state_space_size)
        while np.linalg.norm(state_value_k - self.state_value, ord=1)>tolerance and steps>0:
            steps -= 1
            self.state_value = state_value_k.copy()
            """
                  是普通 policy_improvement 的变种 相当于是值迭代算法 也可以 供策略迭代使用 做策略迭代时不需要 接收第二个返回值
                  更新 qvalue ；qvalue[state,action]=reward+value[next_state]
                  找到 state 处的 action*：action* = arg max(qvalue[state,action]) 即最优action即最大qvalue对应的action
                  更新 policy ：将 action*的概率设为1 其他action的概率设为0 这是一个greedy policy
                  :param: state_value: policy对应的state value
                  :return: improved policy, 以及迭代下一步的state_value
            """
            # 方法初始化了一个新的策略 policy，所有状态的所有动作的概率都被设置为0
            policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            #state_value_k = state_value_k.copy()
            #遍历所有的 state
            q_table = np.zeros(shape=(self.state_space_size, self.action_space_size))
            for state in range(self.state_space_size):
                qvalue_list = []
                #遍历所有的 action
                for action in range(self.action_space_size):
                    # 计算qvalue,即acton value.
                    """
                     计算qvalue elementwise形式
                    :param state: 对应的state
                    :param action: 对应的action
                    :param state_value: 状态值
                    :return: 计算出的结果
                    """
                    qvalue = 0
                    for i in range(self.reward_space_size):
                        # print("self.reward_list[i] * self.env.Rsa[state, action, i]:{}x{}={}".format(self.reward_list[i], self.env.Rsa[state, action, i],self.reward_list[i] * self.env.Rsa[state, action, i]))
                        qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]

                    for next_state in range(self.state_space_size):
                        qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value_k[next_state]
                    qvalue_list.append(qvalue)
                # print("qvalue_list:",qvalue_list)
                q_table[state,:] = qvalue_list.copy()

                state_value_k[state] = max(qvalue_list)  #取该state 的最大state value
                action_star = qvalue_list.index(max(qvalue_list))  #取该state 的最大state value对应的action
                policy[state, action_star] = 1  #更新策略，贪婪算法
            print("q_table:{}".format(q_table))
            self.policy = policy
        return steps


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
    gird_world2x2 = grid_env.GridEnv(size=3, target=[2, 2],
                           forbidden=[[1, 0],[2,1]],
                           render_mode='')

    solver = class_value_iteration(gird_world2x2)
    start_time = time.time()

    # 执行值迭代算法
    demand_step = 1000
    remaining_steps = solver.value_iteration_new(tolerance=0.1, steps=demand_step)
    if remaining_steps > 0:
        print("Value iteration converged in {} steps.".format(demand_step - remaining_steps))
    else:
        print("Value iteration did not converge in 100 steps.")

    end_time = time.time()

    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(gird_world2x2.render_.trajectory))

    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)


    gird_world2x2.render(block=True)