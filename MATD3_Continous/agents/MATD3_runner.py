import numpy as np
import csv
import os
import threading
from datetime import datetime
import copy

class RUNNER:
    def __init__(self, agent, env, par, device, mode = 'evaluate'):
        self.agent = agent
        self.env = env
        self.par = par
        # self.env_agents = {agent_id for agent_id in self.agent.agents.keys()} #  此处创建的是集合，顺序会出问题！ 三个小时花在这里了。。
        # print("self.env_agents:",self.env_agents)  # 此处打印顺序会乱

        self.env_agents = [agent_id  for agent_id in self.agent.agents.keys()]  #将键值 按顺序转换为列表self.env_agents = list(agent_id for agent_id in self.agent.agents.keys())
        self.done = {agent_id : False for agent_id in self.agent.agents.keys()} # 字典
        # print("self.env_agents:",self.env_agents)

        self.best_score = self.par.best_score
        # 添加奖励记录相关的属性
        self.episode_rewards = {}  # 存储每个智能体的详细奖励历史
        self.all_adversary_mean_rewards = []   #添加新的列表来存储每轮 追捕者 的平均奖励

        # 将 agent 的模型放到指定设备上
        for agent in self.agent.agents.values():
            agent.actor.to(device)
            agent.actor_target.to(device)
            agent.critic.to(device)
            agent.critic_target.to(device)

    def train(self):
        step = 0
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.par.episode_num) for agent_id in self.env.agents}
        self.all_adversary_mean_rewards = []  # 追捕者平均奖励记录
        # episode循环
        for episode in range(self.par.episode_num):
            # print(f"This is episode {episode}")
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset(self.par.seed)
            self.done = {agent_id : False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env_agents}

            # 每个智能体与环境进行交互
            while self.env.agents:  #  加入围捕判断
                step += 1  # 此处记录的是并行 的step，即统一执行后，step+1
                if step < self.par.random_steps:
                    action = {agent_id: self.env.action_space(agent_id).sample() for agent_id in self.env.agents}
                else:
                    action = self.agent.select_action(obs, explore=True, total_step=step, noise_type='gaussian') # 使用高斯噪声探索
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                self.agent.add(obs, action, reward, next_obs, self.done)

                # 计算当前episode每个智能体的奖励 每个step求和
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                if step >= self.par.random_steps and step % self.par.learn_interval == 0:
                    # 更新网络
                    self.agent.learn(self.par.batch_size, self.par.gamma) # 目标网络软更新放置在learn函数中，由policy_freq控制更新频率
                # 状态更新 - 创建深拷贝以避免引用问题
                #obs = next_obs
                obs = {k: v.copy() if hasattr(v, 'copy') else copy.deepcopy(v) for k, v in next_obs.items()}

            episode_adversary_rewards = [] # 每轮结束后的记录
            for agent_id, r in agent_reward.items():
                self.episode_rewards[agent_id][episode] = r
                if agent_id.startswith('adversary_'):
                    episode_adversary_rewards.append(r)
            adversary_mean = np.mean(episode_adversary_rewards)
            self.all_adversary_mean_rewards.append(adversary_mean)

            if adversary_mean > self.best_score:
                print(f"New best score,{adversary_mean:>2f},>, {self.best_score:>2f}, saving models...")
                self.agent.save_model(timestamp = False)  #存放在根目录
                self.best_score = adversary_mean
            # 打印进度
            if (episode + 1) % 100 == 0:  # 每100轮打印一次
                message = f'episode {episode + 1}, '
                for agent_id, r in agent_reward.items():
                    message += f'{agent_id}: {r:>4f}; '
                message += f'adversary_mean: {adversary_mean:>4f}'
                print(message)
        # 奖励记录保存为csv
        self.save_rewards_to_csv()   

    def save_rewards_to_csv(self, prefix=''):
        """移植自runner.py的保存方法"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = f"{prefix}rewards_{timestamp}.csv"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(current_dir, '../plot/matd3_data')  # 调整保存路径
        os.makedirs(plot_dir, exist_ok=True)
        
        with open(os.path.join(plot_dir, filename), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Episode'] + list(self.episode_rewards.keys()) + ['Adversary_Mean']
            writer.writerow(header)
            
            for ep in range(self.par.episode_num):
                row = [ep + 1]
                row += [self.episode_rewards[agent_id][ep] for agent_id in self.episode_rewards]
                row.append(self.all_adversary_mean_rewards[ep] if ep < len(self.all_adversary_mean_rewards) else 0)
                writer.writerow(row)
        print(f"Data saved to {os.path.join(plot_dir, filename)}") 

#============================================================================================================
    def evaluate(self):
        """评估训练好的智能体"""
        print("evaluating...")
        # 添加胜率统计变量
        total_episodes = self.par.evaluate_episode_num
        successful_captures = 0
        total_steps = 0
        capture_steps = []

        # 进行多次评估
        for episode in range(self.par.evaluate_episode_num):
            # 初始化环境
            if self.par.use_variable_seeds:
                obs, _ = self.env.reset(self.par.seed + episode)  # 使用不同的种子
            else:
                obs, _ = self.env.reset(self.par.seed)
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env_agents}
            # 记录当前episode的步数
            episode_step = 0
            # 每个智能体与环境进行交互
            while self.env.agents and not any(self.done.values()):
                episode_step += 1
                # 选择动作（评估模式，不添加噪声）
                action = self.agent.select_action(obs, explore=False)
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                self.captured = {agent_id: terminated[agent_id]  for agent_id in self.env_agents}
                captured_flag = any(self.captured.values()) # 捕获成功标志
                if captured_flag:
                    successful_captures += 1
                    capture_steps.append(episode_step)
                    
                # 记录奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                # 更新观测
                obs = {k: v.copy() if hasattr(v, 'copy') else copy.deepcopy(v) for k, v in next_obs.items()}
                # 定期打印奖励信息
                # if episode_step % 10 == 0:  # print_interval
                #     print(f"评估 Episode {episode+1}, 步数: {episode_step}")
                #     for agent_id, r in agent_reward.items():
                #         print(f"{agent_id}: {r:.4f}", end="; ")
                    
                
                # 检查是否达到最大步数
                # if episode_step >= self.par.episode_length:
                #     print(f"评估 Episode {episode+1} 达到最大步数 {self.par.episode_length}")
                #     break
            
            # 计算追捕者平均奖励
            episode_adversary_rewards = []
            for agent_id, r in agent_reward.items():
                if agent_id.startswith('adversary_'):
                    episode_adversary_rewards.append(r)
            adversary_mean = np.mean(episode_adversary_rewards) if episode_adversary_rewards else 0
            
            # # 打印每个评估episode的结果
            # print(f"\n评估 Episode {episode+1} 完成, 总步数: {episode_step}")
            # for agent_id, r in agent_reward.items():
            #     print(f"{agent_id}: {r:.4f}", end="; ")
            # print(f"追捕者平均: {adversary_mean:.4f}")
            
            # # 如果所有智能体都完成了，打印围捕成功
            # if captured_flag:
            #     print(f"围捕成功！用时 {episode_step} 步")
            # total_steps += episode_step
            # print("-" * 50)  # 分隔线
        # 计算并打印胜率统计
        success_rate = successful_captures / total_episodes * 100
        avg_steps = total_steps / total_episodes
        if len(capture_steps) == 0:
            avg_capture_steps = 0
        else:
            avg_capture_steps = sum(capture_steps) / len(capture_steps)
        
        print("\n评估完成")
        print("\n==== 评估统计 ====")
        print(f"总评估轮数: {total_episodes}")
        print(f"成功围捕次数: {successful_captures}")
        print(f"围捕成功率: {success_rate:.2f}%")
        print(f"平均步数/轮: {avg_steps:.2f}")
        print(f"成功围捕平均步数: {avg_capture_steps:.2f}")
        print("=" * 20)
        