import torch
import numpy as np
import argparse
import os
from MAPPO import MAPPO_MPE
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

def setup_seed(seed):
    """Set random seed to ensure reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_env(env_name, ep_len=25, render_mode="human", seed=None, N=1):
    """Create environment and get observation and action dimensions for each agent"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True, render_mode=render_mode)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, N=N, render_mode=render_mode, continuous_actions=True)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode=render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    
    # Handle seed when resetting environment
    if seed is not None:
        new_env.reset(seed=seed)
    else:
        new_env.reset()

    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print(f"Agent ID: {agent_id}")
        _dim_info[agent_id] = []
        action_bound[agent_id] = []
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)
    
    return new_env, _dim_info, action_bound

class MAPPO_Evaluator:
    def __init__(self, args):
        self.args = args
        self.env_name = args.env_name
        self.number = args.number
        self.seed = args.seed
        
        # Set random seed
        setup_seed(self.seed)
        
        # Create environment
        self.env, dim_info, _ = get_env(env_name=self.env_name, 
                                      ep_len=args.episode_limit, 
                                      render_mode=args.render_mode, 
                                      seed=self.seed,
                                      N=self.number)
        
        # 设置参数
        self.args.agents = self.env.agents
        self.args.N = len(self.env.agents)
        self.args.obs_dim_n = [self.env.observation_space(i).shape[0] for i in self.args.agents]
        self.args.action_dim_n = [self.env.action_space(i).shape[0] for i in self.args.agents]
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        print(f"Observation space dimensions: {self.args.obs_dim_n}")
        print(f"Action space dimensions: {self.args.action_dim_n}")
        
        # Create agents
        self.agent_n = MAPPO_MPE(self.args)
        self.agent_n.env = self.env
        
        # Save all agent IDs for later reference
        self.agent_n.all_agents = [agent_id for agent_id in self.env.agents]

        # 加载模型
        if self.args.model_step > 0:
            print(f"加载模型 step {args.model_step}k")
            self.agent_n.load_model(self.env_name, self.number, self.seed, self.args.model_step, self.args.timestamp)
        else:
            print("未指定模型步数或路径，将使用随机策略进行评估")

    
    def evaluate(self):
        """评估训练好的智能体"""
        print("开始评估...")
        total_episodes = self.args.evaluate_episodes
        all_rewards = []
        episode_steps = []
        
        # 进行多次评估
        for episode in range(total_episodes):
            episode_reward, steps = self.run_episode()
            all_rewards.append(episode_reward)
            episode_steps.append(steps)
            print(f"Episode {episode+1}/{total_episodes}: 奖励 = {episode_reward:.4f}, 步数 = {steps}")
        
        # 计算并打印统计信息
        avg_reward = np.mean(all_rewards)
        avg_steps = np.mean(episode_steps)
        print(f"\n===== 评估结果 =====")
        print(f"环境: {self.env_name}")
        print(f"智能体数量: {self.number}")
        print(f"评估回合数: {total_episodes}")
        print(f"平均奖励: {avg_reward:.4f} ± {np.std(all_rewards):.4f}")
        print(f"平均步数: {avg_steps:.2f}")
        print(f"最大奖励: {np.max(all_rewards):.4f}")
        print(f"最小奖励: {np.min(all_rewards):.4f}")
        print("====================")
    
    def run_episode(self):
        """运行一个回合并返回奖励和步数"""
        episode_reward = 0
        obs_dict, _ = self.env.reset()
        
        for episode_step in range(self.args.episode_limit):
            # 根据当前观察选择动作
            actions_dict, _ = self.agent_n.choose_action(obs_dict, evaluate=True)
            
            # 与环境交互
            next_obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions_dict)
            
            # 累计奖励
            step_reward = sum(rewards_dict.values())
            episode_reward += step_reward
            
            # 可视化
            if self.args.render_mode == "human":
                self.env.render()
            
            # 更新观测
            obs_dict = next_obs_dict
            
            # 检查是否结束
            done = any(terminated_dict.values()) or any(truncated_dict.values()) or len(self.env.agents) == 0
            if done:
                break
        
        return episode_reward, episode_step + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MAPPO 评估参数设置")
    parser.add_argument("--env_name", type=str, default="simple_spread_v3", 
                        help="环境名称 (simple_adversary_v3, simple_spread_v3, simple_tag_v3)")
    parser.add_argument("--number", type=int, default=2, 
                        help="智能体数量（仅在simple_spread_v3中有效）")
    parser.add_argument("--seed", type=int, default=23, 
                        help="随机种子")
    parser.add_argument("--timestamp", type=str, default="2025-09-23_20-24", 
                    help="模型时间戳目录，例如：'2025-09-23_01-30'")
    parser.add_argument("--model_step", type=int, default=95, 
                        help="要加载的模型步数（单位：k）")
    parser.add_argument("--episode_limit", type=int, default=50, 
                        help="每回合最大步数")
    parser.add_argument("--evaluate_episodes", type=int, default=10, 
                        help="评估回合数")
    parser.add_argument("--render_mode", type=str, default="human", 
                        help="渲染模式 (human, rgb_array, None)")
    
    # 添加所有MAPPO_MPE初始化所需的参数
    parser.add_argument("--rnn_hidden_dim", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=int(3e6))
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=15)
    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_reward_norm", type=bool, default=False)  # 评估时通常不使用奖励归一化
    parser.add_argument("--use_reward_scaling", type=bool, default=False)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--use_lr_decay", type=bool, default=False)  # 评估时不需要学习率衰减
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=float, default=True)
    parser.add_argument("--use_relu", type=float, default=False)
    parser.add_argument("--use_rnn", type=bool, default=False)
    parser.add_argument("--add_agent_id", type=float, default=False)
    parser.add_argument("--use_value_clip", type=float, default=False)
    parser.add_argument("--act_dim", type=int, default=5)
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = MAPPO_Evaluator(args)
    evaluator.evaluate()