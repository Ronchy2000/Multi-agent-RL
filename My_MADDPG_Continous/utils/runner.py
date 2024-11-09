import numpy as np
import visdom


class RUNNER:
    def __init__(self, agent, env, par):
        self.agent = agent
        self.env = env
        self.par = par

    def train(self):
        # 使用visdom实时查看训练曲线
        viz = None
        if self.par.visdom:
            viz = visdom.Visdom()
            viz.close()
        step = 0
        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        reward_sum_record = []
        # 记录每个智能体在每个episode的奖励
        episode_rewards = {agent_id: np.zeros(self.par.episode_num) for agent_id in self.env.agents}
        # episode循环
        for episode in range(self.par.episode_num):
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset()
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            # 每个智能体与环境进行交互
            while self.env.agents:
                step += 1
                # 未到学习阶段 所有智能体随机选择动作 动作同样为字典 键为智能体名字 值为对应的动作 这里为随机选择动作
                if step < self.par.random_steps:
                    action = {agent_id: self.env.action_space(agent_id).sample() for agent_id in self.env.agents}
                # 开始学习 根据策略选择动作
                else:
                    action = self.agent.select_action(obs)
                # 执行动作 获得下一状态 奖励 终止情况
                # 下一状态：字典 键为智能体名字 值为对应的下一状态
                # 奖励：字典 键为智能体名字 值为对应的奖励
                # 终止情况：bool
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                # env.render()
                # 存储样本
                self.agent.add(obs, action, reward, next_obs, done)
                # 计算当前episode每个智能体的奖励 每个step求和
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                # 开始学习 有学习开始条件 有学习频率
                if step >= self.par.random_steps and step % self.par.learn_interval == 0:
                    # 学习
                    self.agent.learn(self.par.batch_size, self.par.gamma)
                    # 更新网络
                    self.agent.update_target(self.par.tau)
                # 状态更新
                obs = next_obs
            # 记录、绘制每个智能体在当前episode中的和奖励
            sum_reward = 0
            for agent_id, r in agent_reward.items():
                sum_reward += r
                if self.par.visdom:
                    viz.line(X=[episode + 1], Y=[r], win='sum reward of the agent ' + str(agent_id),
                             opts={'title': 'reward of the agent ' + str(agent_id) + ' in all episode'},
                             update='append')
            # 绘制所有智能体在当前episode的和奖励
            if self.par.visdom:
                viz.line(X=[episode + 1], Y=[sum_reward], win='Sum reward of all agents',
                         opts={'title': 'Sum reward of all agents in all episode'},
                         update='append')
            # 记录当前episode的所有智能体和奖励 为奖励平滑做准备
            reward_sum_record.append(sum_reward)
            # 保存当前智能体在当前episode的奖励
            for agent_id, r in agent_reward.items():
                episode_rewards[agent_id][episode] = r
            # 根据平滑窗口确定打印间隔 并进行平滑
            if (episode + 1) % self.par.size_win == 0:
                message = f'episode {episode + 1}, '
                sum_reward = 0
                for agent_id, r in agent_reward.items():
                    message += f'{agent_id}: {r:>4f}; '
                    sum_reward += r
                message += f'sum reward: {sum_reward}'
                print(message)
                if self.par.visdom:
                    epi = np.linspace(episode - (self.par.size_win - 2),
                                      episode - (self.par.size_win - 2) + (self.par.size_win - 1), self.par.size_win,
                                      dtype=int)
                    viz.line(X=epi, Y=self.get_running_reward(reward_sum_record), win='Average sum reward',
                             opts={'title': 'Average sum reward'},
                             update='append')
                reward_sum_record = []

    def get_running_reward(self, arr):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        window = self.par.size_win
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward

    @staticmethod
    def exponential_moving_average(rewards, alpha=0.1):
        """计算指数移动平均奖励"""
        ema_rewards = np.zeros_like(rewards)
        ema_rewards[0] = rewards[0]
        for t in range(1, len(rewards)):
            ema_rewards[t] = alpha * rewards[t] + (1 - alpha) * ema_rewards[t - 1]
        return ema_rewards

    import numpy as np

    def moving_average(self, rewards):
        """计算简单移动平均奖励"""
        window_size = self.par.size_win
        sma_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        return sma_rewards
