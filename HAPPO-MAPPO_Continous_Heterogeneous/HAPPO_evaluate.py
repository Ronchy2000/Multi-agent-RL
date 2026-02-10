import torch
import numpy as np
import argparse
import os
from HAPPO import HAPPO_MPE
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

def get_env(env_name, ep_len=25, render_mode="human", seed=None, num_good=1, num_adversaries=3):
    """Create environment and get observation and action dimensions for each agent"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True, render_mode=render_mode)
    elif env_name == 'simple_spread_v3':
        # 对于simple_spread_v3，使用N参数
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, N=num_good, render_mode=render_mode, continuous_actions=True)
    elif env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode=render_mode, num_good=num_good, num_adversaries=num_adversaries, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    
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
    
    print(f"Environment agent dimension info: {_dim_info}")
    return new_env, _dim_info, action_bound

class HAPPO_Evaluator:
    def __init__(self, args):
        self.args = args
        self.env_name = args.env_name
        self.seed = args.seed
        
        # Set random seed
        setup_seed(self.seed)
        
        # Create environment - use different parameters based on environment type
        if self.env_name == 'simple_tag_v3':
            # simple_tag_v3 environment uses num_good and num_adversaries
            self.env, self.dim_info, _ = get_env(
                env_name=self.env_name, 
                ep_len=args.episode_limit, 
                render_mode=args.render_mode, 
                seed=self.seed,
                num_good=args.num_good,
                num_adversaries=args.num_adversaries
            )
        else:
            # Other environments use number parameter
            self.env, self.dim_info, _ = get_env(
                env_name=self.env_name, 
                ep_len=args.episode_limit, 
                render_mode=args.render_mode, 
                seed=self.seed,
                num_good=args.number
            )
        
        # Set environment parameters
        self.args.agents = self.env.agents
        self.args.N = len(self.env.agents)
        self.args.obs_dim_n = [self.env.observation_space(i).shape[0] for i in self.args.agents]
        self.args.action_dim_n = [self.env.action_space(i).shape[0] for i in self.args.agents]
        
        # 为异构环境设置正确的维度
        self.max_obs_dim = max(self.args.obs_dim_n)
        self.args.obs_dim = self.max_obs_dim  # 使用最大观察维度
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        print(f"智能体列表: {self.args.agents}")
        print(f"观察空间维度: {self.args.obs_dim_n}")
        print(f"动作空间维度: {self.args.action_dim_n}")
        print(f"最大观察维度: {self.max_obs_dim}")
        print(f"全局状态维度: {self.args.state_dim}")
        
        # 创建HAPPO智能体
        self.agent_n = HAPPO_MPE(self.args)
        self.agent_n.env = self.env
        self.agent_n.all_agents = [agent_id for agent_id in self.env.agents]
        self.agent_n.dim_info = self.dim_info  # 设置维度信息
        
        # 为环境添加get_obs_dims方法
        def get_obs_dims():
            return {agent_id: self.env.observation_space(agent_id).shape[0] for agent_id in self.env.agents}
        self.env.get_obs_dims = get_obs_dims

        # 加载模型
        if self.args.model_step > 0:
            print(f"加载模型 step {args.model_step}k，时间戳: {args.timestamp}")
            self.load_hetero_model()
        else:
            print("未指定模型步数，将使用随机策略进行评估")

    def load_hetero_model(self):
        """加载异构模型"""
        try:
            # 构建模型路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, "models")
            
            if self.args.timestamp:
                timestamp_dir = os.path.join(models_dir, self.args.timestamp)
            else:
                timestamp_dir = models_dir
                
            # 对于simple_tag_v3，使用number=1 (因为训练时使用的是这个值)
            if self.env_name == 'simple_tag_v3':
                number = 1
            else:
                number = self.args.number
                
            actor_path = os.path.join(
                timestamp_dir, 
                f"HAPPO_actor_env_{self.env_name}_number_{number}_seed_{self.seed}_step_{self.args.model_step}k.pth"
            )
            
            print(f"尝试加载模型: {actor_path}")
            
            if not os.path.exists(actor_path):
                raise FileNotFoundError(f"模型文件不存在: {actor_path}")
            
            # 加载模型文件
            data = torch.load(actor_path, map_location=self.args.device)
            
            # 检查模型格式
            if isinstance(data, dict) and 'actor_state_dict_by_agent' in data:
                print(f"检测到异构模型格式: {data.get('format', 'unknown')}")
                print(f"模型包含的智能体: {data.get('agents', [])}")
                
                # 确保异构智能体已构建
                self.agent_n._build_hetero_if_needed()
                
                # 加载每个智能体的Actor
                loaded_agents = []
                for agent_id, state_dict in data['actor_state_dict_by_agent'].items():
                    if agent_id in self.agent_n.agents:
                        self.agent_n.agents[agent_id].actor.load_state_dict(state_dict)
                        loaded_agents.append(agent_id)
                        print(f"成功加载智能体 {agent_id} 的Actor模型")
                    else:
                        print(f"智能体 {agent_id} 不在当前环境中，跳过加载")
                
                print(f"总共加载了 {len(loaded_agents)} 个智能体的模型")
                
                # 验证是否所有当前环境的智能体都有对应的模型
                missing_agents = set(self.agent_n.all_agents) - set(loaded_agents)
                if missing_agents:
                    print(f"以下智能体缺少训练好的模型: {missing_agents}")
                    print("这些智能体将使用随机初始化的策略")
                
            else:
                print("检测到旧格式模型，将复制到所有智能体")
                self.agent_n._build_hetero_if_needed()
                for agent_id in self.agent_n.agents:
                    self.agent_n.agents[agent_id].actor.load_state_dict(data)
                print("已将共享模型复制到所有智能体")
            
            print("模型加载完成!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用随机初始化的策略进行评估")
    
    def evaluate(self):
        """评估训练好的智能体"""
        print("\n" + "="*50)
        print("开始评估...")
        print("="*50)
        
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
        
        print("\n" + "="*50)
        print("评估结果")
        print("="*50)
        print(f"环境: {self.env_name}")
        print(f"智能体列表: {self.args.agents}")
        print(f"评估回合数: {total_episodes}")
        print(f"平均奖励: {avg_reward:.4f} ± {np.std(all_rewards):.4f}")
        print(f"平均步数: {avg_steps:.2f}")
        print(f"最大奖励: {np.max(all_rewards):.4f}")
        print(f"最小奖励: {np.min(all_rewards):.4f}")
        print("="*50)
    
    def run_episode(self):
        """运行一个回合并返回奖励和步数"""
        episode_reward = 0
        obs_dict, _ = self.env.reset()
        
        # 如果使用RNN，重置所有智能体的隐藏状态
        if self.args.use_rnn:
            self.agent_n.reset_rnn_hidden()
        
        for episode_step in range(self.args.episode_limit):
            # 根据当前观察选择动作（evaluate=True表示使用确定性策略）
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
    parser = argparse.ArgumentParser("HAPPO 评估参数设置")
    parser.add_argument("--env_name", type=str, default="simple_tag_v3", 
                        help="环境名称 (simple_adversary_v3, simple_spread_v3, simple_tag_v3)")
    parser.add_argument("--number", type=int, default=1, 
                        help="智能体数量（仅在simple_spread_v3等环境中有效）")
    parser.add_argument("--num_good", type=int, default=1, 
                        help="逃跑者数量（simple_tag_v3环境）")
    parser.add_argument("--num_adversaries", type=int, default=3, 
                        help="追捕者数量（simple_tag_v3环境）")
    parser.add_argument("--seed", type=int, default=23, 
                        help="随机种子")
    parser.add_argument("--timestamp", type=str, default="2025-09-24_20-12", 
                    help="模型时间戳目录，例如：'2025-09-24_20-20'")
    parser.add_argument("--model_step", type=int, default=410, 
                        help="要加载的模型步数（单位：k）")
    parser.add_argument("--episode_limit", type=int, default=50, 
                        help="每回合最大步数")
    parser.add_argument("--evaluate_episodes", type=int, default=5, 
                        help="评估回合数")
    parser.add_argument("--render_mode", type=str, default="human", 
                        help="渲染模式 (human, rgb_array, None)")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="计算设备")
    
    # 添加所有HAPPO_MPE初始化所需的参数
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
    evaluator = HAPPO_Evaluator(args)
    evaluator.evaluate()