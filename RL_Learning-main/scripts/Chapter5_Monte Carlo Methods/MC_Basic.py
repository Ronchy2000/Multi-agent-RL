import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
sys.path.append("..")
import grid_env

from tqdm import tqdm


'''
MC Basic 是个Model free 的方法，与value iteration和 Policy iteration对比，数据是MC的必需品。
'''
class MC_Basic:
    def __init__(self, env = grid_env.GridEnv):
        self.gama = 0.9   #discount rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list  # [-10,-10,0,1]  reward list
        self.state_value = np.zeros(shape=self.state_space_size) #一维列表
        print("self.state_value:",self.state_value)
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
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})  #向列表中添加一个字典
        return episode  #返回列表，其中的元素为字典


    """
    jkrose作者这个代码不是MC_basic代码，而是MC_exploring_start every visit形式的代码。
    MC_Basic算法请查看mc_basic_simple() 函数
    """
    def mc_basic(self, length=50, epochs=10):
        """
        :param length: 每一个 state-action 对的长度
        :return:
        """
        for epoch in range(epochs):
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(self.policy, state, action, length)
                    g = 0
                    print("obtain_episode,type:,{}; {}".format(type(episode[0]), episode))
                    # 这里原作者利用递归的思想求qvalue,实际上可以傻瓜式求解。
                    for step in range(len(episode)-1, -1, -1):
                        g = episode[step]['reward'] + self.gama * g
                    self.qvalue[state][action] = g

                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                self.policy[state] = np.zeros(shape=self.action_space_size)
                self.policy[state, action_star] = 1
                self.state_value[state] = qvalue_star.copy()
            print(epoch)
        return self.state_value
    """
    mc_basic_simple函数与书中的MC_basic算法伪代码一致。  2024.5.15
    return: self.state_value 为了render结果图，即可视化
    """
    def mc_basic_simple(self, length=20, epochs=10):
        """
        :param length: 每一个 state-action 对的长度
        :return:
        """
        num_episode = 5
        episodes = []
        for epoch in range(epochs):
            for state in range(self.state_space_size):
                """
                Page97: The first strategy is, in the policy evaluation step, 
                to collect all the episodes starting from the same state-action pair and then approximate the action 
                value using the average return of these episodes. This strategy is adopted in the MC Basic algorithm.
                我的理解：对于一个state，获取到了该状态下，每一个action下的episode之后再进行 policy improvement
                """
                for action in range(self.action_space_size):
                    # Collect sufficiently many episodes starting from (s, a) by following πk
                    for tmp in range(num_episode): #对每个action 采集 10条 episode
                        episodes.append(self.obtain_episode(self.policy, state, action, length))
                    #Policy evaluation:
                    sum_qvalue_list = []
                    for each_episode in episodes:
                        sum_qvalue = 0
                        for i in range(len(each_episode)):
                            sum_qvalue += (self.gama**i) * each_episode[i]['reward']
                    sum_qvalue_list.append(sum_qvalue)
                    self.qvalue[state][action] = np.mean(sum_qvalue_list) #the average return of all the episodes starting from (s, a)
                    # self.qvalue[state][action] = np.sum(sum_qvalue_list)/num_episode
                #Policy improvement:
                max_index = np.argmax(self.qvalue[state])
                max_qvalue = np.max(self.qvalue[state])
                self.policy[state,:] = np.zeros(self.action_space_size)
                self.policy[state,max_index] = 1
                self.state_value[state] = max_qvalue
            print("epoch:", epoch)
        return self.state_value


    def mc_basic_simple_GUI(self, length=50, epochs=10):
        """
        :param length: 每一个 state-action 对的长度
        """
        num_episode = 5
        episodes = []
        for epoch in range(epochs):
            for state in tqdm(range(self.state_space_size), desc = f"Epoch {epoch}/{epochs}"):
                for action in range(self.action_space_size):
                    # Collect sufficiently many episodes starting from (s, a) by following πk
                    for tmp in range(num_episode):  # 对每个action 采集 10条 episode
                        episodes.append(self.obtain_episode(self.policy, state, action, length))
                    # Policy evaluation:
                    sum_qvalue_list = []
                    for each_episode in episodes:
                        sum_qvalue = 0
                        for i in range(len(each_episode)):
                            sum_qvalue += (self.gama ** i) * each_episode[i]['reward']
                    sum_qvalue_list.append(sum_qvalue)
                    self.qvalue[state][action] = np.mean(
                        sum_qvalue_list)  # the average return of all the episodes starting from (s, a)
                    # self.qvalue[state][action] = np.sum(sum_qvalue_list)/num_episode
                # Policy improvement:
                max_index = np.argmax(self.qvalue[state])
                max_qvalue = np.max(self.qvalue[state])
                self.policy[state, :] = np.zeros(self.action_space_size)
                self.policy[state, max_index] = 1
                self.state_value[state] = max_qvalue




if __name__ == "__main__":
    episode_length = [20]
    # episode_length = [1,2,3,4,14,15,30,100]
    # episode_length = [1, 2, 3, 4]
    for i in episode_length:
        gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                      forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                      render_mode='')
        solver = MC_Basic(gird_world)
        start_time = time.time()

        # solver.state_value = solver.mc_basic_simple(length=i, epochs=10)
        solver.mc_basic_simple_GUI(length=i, epochs=10)  #修改后，利用tqdm显示epoch进度

        end_time = time.time()
        cost_time = end_time - start_time
        print("episode_length:{} that the cost_time is:{}".format(i,round(cost_time, 2)))

        print(len(gird_world.render_.trajectory))
        solver.show_policy()  # solver.env.render()
        solver.show_state_value(solver.state_value, y_offset=0.25)
        gird_world.plot_title("Episode_length = "+str(i))
        gird_world.render()
        # gird_world.render_clear()
        print("--------------------")




"""
tqdm库测试
"""
if __name__ == "__main1__":
    # 记录开始时间
    start_time = time.time()

    # 迭代过程
    for i in tqdm(range(100)):
        time.sleep(0.1)  # 模拟一些耗时的操作

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time

    # 在 tqdm 进度条描述中显示时间信息
    print("Algorithm finished in {:.2f} seconds.".format(elapsed_time))
