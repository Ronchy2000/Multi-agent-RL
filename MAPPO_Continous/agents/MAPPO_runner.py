import sys
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from MAPPO_agent import MAPPO_MPE
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
import copy
import csv
from datetime import datetime

class Runner_MAPPO_MPE:
    def __init__(self, agent, env, args, device, mode='train'):
        """
        初始化MAPPO运行器
        
        参数:
            agent: MAPPO智能体
            env: PettingZoo并行环境
            args: 参数对象
            device: 训练设备
            mode: 'train'或'evaluate'
        """
        self.agent = agent
        self.env = env
        self.args = args
        self.device = device
        self.mode = mode
        
        # 存储环境中的智能体ID (不直接使用env.agents，因为在PettingZoo中智能体可能会被移除)
        self.env_agents = [agent_id for agent_id in self.env.agents]
        self.done = {agent_id: False for agent_id in self.env_agents}
        
        # 奖励记录
        self.episode_rewards = {}  # 存储每个智能体的详细奖励历史
        self.all_adversary_mean_rewards = []  # 追捕者平均奖励
        self.evaluate_rewards = []  # 评估奖励记录

        # 全局步数计数
        self.total_steps = 0

        # 设置tensorboard (可选)
        if mode == 'train' and hasattr(args, 'use_tensorboard') and args.use_tensorboard:
            self.writer = SummaryWriter(log_dir=f'runs/MAPPO/{args.env_name}_{args.seed}')
        else:
            self.writer = None
            
        # 设置奖励处理工具
        # if args.use_reward_norm:
        #     print("使用奖励规范化")
        #     self.reward_norm = Normalization(shape=len(self.env_agents))
        # elif args.use_reward_scaling:
        #     print("使用奖励缩放")
        #     self.reward_scaling = RewardScaling(shape=len(self.env_agents), gamma=args.gamma)
        # 初始化奖励归一化
        if self.args.use_reward_norm:
            print("使用奖励规范化")
            # 确保奖励归一化使用正确的形状 - 应该是智能体数量
            self.reward_norm = Normalization(shape=len(self.env_agents))
        else:
            self.reward_norm = None
    
    def run(self):
        """
        运行完整的训练流程
        """
        print("开始MAPPO训练流程...")
        evaluate_num = -1  # 记录评估次数
        
        # 初始化回放缓冲区
        self.replay_buffer = ReplayBuffer(self.args)
        
        # 初始化奖励记录
        self.episode_rewards = {agent_id: [] for agent_id in self.env_agents}
        self.evaluate_rewards = []
        
        while self.total_steps < self.args.episode_num:
            # 定期评估策略
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # 每evaluate_freq步评估一次策略
                evaluate_num += 1
            
            # 运行一个回合收集数据
            _, episode_steps = self.run_episode_mpe(evaluate=False)
            self.total_steps += episode_steps
            
            # 达到批次大小后进行训练
            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent.train(self.replay_buffer, self.total_steps)  # 训练
                self.replay_buffer.reset_buffer()  # 重置缓冲区
        
        # 训练结束后的最终评估
        self.evaluate_policy()
        
        # 关闭环境
        self.env.close()
        
        print(f"MAPPO训练完成! 总步数: {self.total_steps}")
        return self.episode_rewards, self.all_adversary_mean_rewards

    def evaluate_policy(self):
        """
        评估当前策略
        
        返回:
            mean_reward: 平均奖励
        """
        print("正在评估策略...")
        evaluate_reward = 0
        
        # 进行多次评估
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            # 计算所有智能体的总奖励
            episode_total_reward = sum(episode_reward.values())
            evaluate_reward += episode_total_reward
        
        # 计算平均奖励
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        
        # 输出评估结果
        print(f"total_steps:{self.total_steps} \t evaluate_reward:{evaluate_reward:.4f}")
        
        # 如果使用tensorboard，记录评估结果
        if self.writer is not None:
            self.writer.add_scalar(f'evaluate_step_rewards_{self.args.env_name}', 
                                evaluate_reward, global_step=self.total_steps)
        
        # 保存奖励和模型
        os.makedirs('./data_train', exist_ok=True)
        np.save(f'./data_train/MAPPO_env_{self.args.env_name}_seed_{self.args.seed}.npy', 
            np.array(self.evaluate_rewards))
        
        # 保存模型
        self.agent.save_model(timestamp=True)
        
        return evaluate_reward

    # 在run_episode_mpe方法中更新处理奖励和动作的代码
    def run_episode_mpe(self, evaluate=False):
        """运行一个回合"""
        # 初始化回合奖励
        episode_reward = {agent_id: 0 for agent_id in self.env_agents}
        
        # 重置环境
        obs, _ = self.env.reset(seed=self.args.seed if not self.args.use_variable_seeds else 
                            self.args.seed + len(self.episode_rewards[list(self.env_agents)[0]]))
        
        # 重置完成状态
        self.done = {agent_id: False for agent_id in self.env_agents}
        
        # 重置奖励缩放
        if hasattr(self, 'reward_scaling') and self.args.use_reward_scaling:
            self.reward_scaling.reset()
        
        # 重置RNN隐藏状态
        if self.args.use_rnn:
            for actor in self.agent.actors.values():
                if hasattr(actor, 'rnn_hidden'):
                    actor.rnn_hidden = None
            if hasattr(self.agent.critic, 'rnn_hidden'):
                self.agent.critic.rnn_hidden = None
        
        # 运行一个回合
        episode_step = 0
        while episode_step < self.args.episode_length and self.env.agents and not all(self.done.values()):
            # 选择动作
            a_n, a_logprob_n = self.agent.choose_action(obs, evaluate, self.env_agents)
            scale_factor = 100.0  # 调整此值控制动作大小
            a_n = a_n * scale_factor
            # 计算全局状态
            s = np.concatenate([obs[agent_id] for agent_id in self.env_agents])
            
            # 获取值函数
            v_n = None if evaluate else self.agent.get_value(s)
            
            # 准备动作字典
            actions = {agent_id: a_n[i] for i, agent_id in enumerate(self.env_agents) if agent_id in self.env.agents}
            print("action:", a_n)
            # 执行动作
            next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
            
            # 更新完成状态
            for agent_id in self.env_agents:
                if agent_id in terminated:
                    self.done[agent_id] = terminated[agent_id] or truncated.get(agent_id, False)
            
            # 记录奖励
            for agent_id in self.env_agents:
                if agent_id in rewards:
                    episode_reward[agent_id] += rewards[agent_id]
            
            # 存储经验（训练模式）
            if not evaluate:
                # 准备数据
                r_n = np.array([rewards.get(agent_id, 0.0) for agent_id in self.env_agents])
                done_n = np.array([self.done.get(agent_id, False) for agent_id in self.env_agents])
                
                # 奖励处理
                if self.reward_norm is not None:
                    r_n = self.reward_norm(r_n, update=True)
                elif hasattr(self, 'reward_scaling') and self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
                    
                # 存储经验
                self.replay_buffer.store_transition(episode_step, obs, s, v_n, a_n, a_logprob_n, r_n, done_n)
            
            # 更新状态
            obs = next_obs
            episode_step += 1
        
        # 存储最后一步的值函数（训练模式）
        if not evaluate:
            # 计算最后一个状态的值函数
            if self.env.agents:  # 如果环境中还有智能体
                s = np.concatenate([obs[agent_id] for agent_id in self.env_agents if agent_id in obs])
                v_n = self.agent.get_value(s)
            else:  # 如果环境中没有智能体（所有智能体都已完成）
                v_n = np.zeros(len(self.env_agents))
            
            # 存储最后一步的值函数
            self.replay_buffer.store_last_value(episode_step, v_n)
        
        return episode_reward, episode_step

    def save_rewards_to_csv(self, prefix=''):
        """
        保存奖励记录到CSV文件
        
        参数:
            prefix: 文件名前缀
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = f"{prefix}mappo_rewards_{timestamp}.csv"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(current_dir, '../plot/mappo_data')
        os.makedirs(plot_dir, exist_ok=True)
        
        # 确保所有列表长度一致
        max_length = max([len(rewards) for rewards in self.episode_rewards.values()] +
                        [len(self.all_adversary_mean_rewards)])
        
        with open(os.path.join(plot_dir, filename), 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            header = ['Episode'] + list(self.episode_rewards.keys()) + ['Adversary_Mean']
            writer.writerow(header)
            
            # 写入数据
            for ep in range(max_length):
                row = [ep + 1]
                
                # 添加每个智能体的奖励
                for agent_id in self.episode_rewards:
                    rewards = self.episode_rewards[agent_id]
                    row.append(rewards[ep] if ep < len(rewards) else None)
                
                # 添加追捕者平均奖励
                row.append(self.all_adversary_mean_rewards[ep] if ep < len(self.all_adversary_mean_rewards) else None)
                
                writer.writerow(row)
                
        print(f"奖励数据已保存到 {os.path.join(plot_dir, filename)}")

    def evaluate(self, n_episodes=10):
        """
        评估当前策略的性能
        
        参数:
            n_episodes: 要评估的回合数
            
        返回:
            capture_rate: 捕获成功率
            avg_capture_steps: 平均捕获步数
        """
        if self.mode != 'evaluate':
            print("警告: 在非评估模式下调用evaluate()方法")
        
        print(f"评估{n_episodes}个回合...")
        
        captures = 0  # 成功捕获次数
        capture_steps = []  # 每次捕获所需步数
        
        for ep in range(n_episodes):
            # 运行一个回合
            episode_reward, steps = self.run_episode_mpe(evaluate=True)
            # 检查是否捕获 - 在simple_tag环境中
            # 这里使用一个简单的启发式方法：如果逃跑者(agent_0)奖励足够低，认为被捕获
            agent_0_reward = [reward for agent_id, reward in episode_reward.items() 
                            if not agent_id.startswith('adversary_')]
            
            if agent_0_reward and agent_0_reward[0] < -15:  # 调整阈值可能需要根据环境实际情况
                captures += 1
                capture_steps.append(steps)
        
        # 计算统计指标
        capture_rate = captures / n_episodes if n_episodes > 0 else 0
        avg_capture_steps = np.mean(capture_steps) if capture_steps else self.args.episode_length
        
        print(f"评估完成! 成功率: {capture_rate:.4f}, 平均步数: {avg_capture_steps:.2f}")
        
        return capture_rate, avg_capture_steps

# 如果直接运行此文件
if __name__ == '__main__':
    print("请通过main_train.py运行MAPPO训练")