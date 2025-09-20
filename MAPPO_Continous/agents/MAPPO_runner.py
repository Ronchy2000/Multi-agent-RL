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
import os

class Runner_MAPPO_MPE:
    def __init__(self, agent, env, args, device, mode = 'train'):
        self.agent = agent
        self.env = env
        self.args = args
        self.device = device
        self.mode = mode
        
        # 这里为什么新建而不是直接使用agent.agents.keys()？
        # 因为pettingzoo中智能体死亡，这个字典就没有了，会导致更新出错
        self.env_agents = [agent_id for agent_id in self.env.agents]
        self.done = {agent_id: False for agent_id in self.env_agents}
        
        # 添加奖励记录相关的属性
        self.episode_rewards = {}  # 存储每个智能体的详细奖励历史
        self.all_adversary_mean_rewards = []  # 追捕者平均奖励

        # 设置tensorboard
        if mode == 'train':
            self.writer = SummaryWriter(log_dir=f'runs/MAPPO/{args.env_name}_{args.seed}')
        # 设置规范化工具
        if args.use_reward_norm:
            print("使用奖励规范化")
            self.reward_norm = Normalization(shape=len(self.env_agents))
        elif args.use_reward_scaling:
            print("使用奖励缩放")
            self.reward_scaling = RewardScaling(shape=len(self.env_agents), gamma=args.gamma)
    
    def train(self):
        print("开始训练...")
        total_steps = 0
        self.episode_rewards = {agent_id: np.zeros(self.args.episode_num) for agent_id in self.env.agents}
        # 创建经验回放缓冲区
        replay_buffer = ReplayBuffer(self.args)
        # 开始训练循环
        for episode in range(self.args.episode_num):
            # 重置环境
            if self.args.use_variable_seeds:
                obs, _ = self.env.reset(self.args.seed + episode)
            else:
                obs, _ = self.env.reset(self.args.seed)
            self.done = {agent_id: False for agent_id in self.env_agents}
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            # 如果使用奖励缩放，重置缩放器
            if self.args.use_reward_scaling:
                self.reward_scaling.reset()
            # 一个回合的循环
            for episode_step in range(self.args.episode_length):
                # 选择动作
                a_n, a_logprob_n = self.agent.choose_action(list(obs.values()), evaluate=False)
                # 获取全局状态
                s = np.concatenate(list(obs.values()), axis=0)     
                # 获取值函数估计
                v_n = self.agent.get_value(s)
                # 执行动作
                # 将a_n转换为字典
                action_dict = {}
                for i, agent_id in enumerate(self.env.agents):
                    action_dict[agent_id] = a_n[i]
                # 执行动作
                next_obs, reward, terminated, truncated, _ = self.env.step(action_dict)
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                # 整理奖励
                r_n = np.array(list(reward.values()))
                done_n = np.array([self.done[agent_id] for agent_id in self.env_agents])
                # 可选：对奖励进行规范化处理
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
                # 存储转换
                replay_buffer.store_transition(
                    episode_step, 
                    list(obs.values()), 
                    s, 
                    v_n, 
                    a_n, 
                    a_logprob_n, 
                    r_n, 
                    done_n
                )
                # 累积奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                # 更新观测
                obs = copy.deepcopy(next_obs)
                # 如果所有智能体都完成，跳出循环
                if all(self.done.values()):
                    break
            # 回合结束，存储最后的值函数
            s = np.concatenate(list(obs.values()), axis=0)
            v_n = self.agent.get_value(s)
            replay_buffer.store_last_value(episode_step + 1, v_n)
            # 记录每个智能体的奖励
            for agent_id, r in agent_reward.items():
                self.episode_rewards[agent_id][episode] = r
            # 计算追捕者平均奖励
            adversary_rewards = []
            for agent_id, r in agent_reward.items():
                if agent_id.startswith('adversary_'):
                    adversary_rewards.append(r)
            adversary_mean = np.mean(adversary_rewards) if adversary_rewards else 0
            self.all_adversary_mean_rewards.append(adversary_mean)
            # 每隔一定回合训练一次
            if replay_buffer.episode_num >= self.args.batch_size:
                self.agent.train(replay_buffer, total_steps)
                replay_buffer.reset_buffer()
            # 输出训练进度
            if (episode + 1) % 10 == 0:
                message = f'Episode {episode + 1}/{self.args.episode_num}, '
                for agent_id, r in agent_reward.items():
                    message += f'{agent_id}: {r:.4f}; '
                message += f'adversary_mean: {adversary_mean:.4f}'
                print(message)
            total_steps += episode_step + 1
        # 保存模型和奖励记录
        self.save_rewards_to_csv()
        return self.episode_rewards, self.all_adversary_mean_rewards

    def evaluate(self):
        """评估训练好的智能体"""
        print("正在评估...")
        # 添加统计变量
        successful_captures = 0
        total_steps = 0
        capture_steps = []
        # 进行多次评估
        for episode in range(self.args.evaluate_episode_num):
            # 重置环境
            if self.args.use_variable_seeds:
                obs, _ = self.env.reset(self.args.seed + episode)
            else:
                obs, _ = self.env.reset(self.args.seed)
            self.done = {agent_id: False for agent_id in self.env_agents}
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            episode_step = 0
            # 每个智能体与环境交互
            while self.env.agents and not any(self.done.values()):
                episode_step += 1
                # 选择动作(评估模式)
                a_n, _ = self.agent.choose_action(list(obs.values()), evaluate=True)
                # 将a_n转换为字典
                action_dict = {}
                for i, agent_id in enumerate(self.env.agents):
                    action_dict[agent_id] = a_n[i]
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action_dict)
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                # 检查是否有捕获成功
                captured = {agent_id: terminated[agent_id] for agent_id in self.env_agents}
                captured_flag = any(captured.values())
                if captured_flag:
                    successful_captures += 1
                    capture_steps.append(episode_step)
                # 记录奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                # 更新观测
                obs = copy.deepcopy(next_obs)
                # 检查是否达到最大步数
                if episode_step >= self.args.episode_length:
                    break
            total_steps += episode_step
        # 计算捕获率和平均捕获步数
        capture_rate = successful_captures / self.args.evaluate_episode_num
        avg_capture_steps = np.mean(capture_steps) if capture_steps else 0
        print(f"评估完成: 总场次={self.args.evaluate_episode_num}, 捕获成功率={capture_rate:.2f}, 平均捕获步数={avg_capture_steps:.2f}")
        return capture_rate, avg_capture_steps

    def save_rewards_to_csv(self, prefix=''):
        """保存奖励记录到CSV文件"""
        # 实现类似于MATD3_runner.py中的保存方法
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = f"{prefix}mappo_rewards_{timestamp}.csv"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(current_dir, '../plot/mappo_data')
        os.makedirs(plot_dir, exist_ok=True)
        with open(os.path.join(plot_dir, filename), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Episode'] + list(self.episode_rewards.keys()) + ['Adversary_Mean']
            writer.writerow(header)
            for ep in range(self.args.episode_num):
                row = [ep + 1]
                row += [self.episode_rewards[agent_id][ep] for agent_id in self.episode_rewards]
                row.append(self.all_adversary_mean_rewards[ep] if ep < len(self.all_adversary_mean_rewards) else 0)
                writer.writerow(row)
        print(f"数据已保存到 {os.path.join(plot_dir, filename)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=3, seed=23)
    runner.run()
