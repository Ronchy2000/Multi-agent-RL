from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from main_parameters import main_parameters

# 修改导入路径
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.MATD3_runner import RUNNER
from agents.MATD3_agent import MATD3
import torch
import random
import numpy as np
from envs import simple_tag_env
import imageio  # 需要安装: pip install imageio

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_env(env_name, ep_len=50, render_mode="None", seed=None):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode=render_mode)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode=render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)    
    if env_name == 'simple_tag_env':
        new_env = simple_tag_env.parallel_env(render_mode=render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)

    new_env.reset(seed=seed)
    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:", agent_id)
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = [] #[low action, high action]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)

    return new_env, _dim_info, action_bound

# 修改RUNNER类以捕获帧
class RecordingRunner(RUNNER):
    def evaluate(self):
        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        self.reward_sum_record = []
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.par.episode_num) for agent_id in self.env.agents}
        frames = []  # 用于存储渲染帧
        
        # episode循环
        for episode in range(self.par.episode_num):
            step = 0  # 每回合step重置
            print(f"评估第 {episode + 1} 回合")
            # 初始化环境 返回初始状态
            obs, _ = self.env.reset(seed=self.par.seed)  # 重置环境，开始新回合
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            
            # 捕获初始帧
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)
            
            # 每个智能体与环境进行交互
            while self.env.agents:
                step += 1
                # 使用训练好的智能体选择动作
                action = self.agent.select_action(obs)
                # 执行动作 获得下一状态 奖励 终止情况
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 捕获当前帧
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)
                
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                # 累积每个智能体的奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                obs = next_obs
                if step % 10 == 0:
                    print(f"Step {step}, action: {action}, reward: {reward}, done: {self.done}")
            
            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)
            print(f"回合 {episode + 1} 总奖励: {sum_reward}")
        
        return frames

if __name__ == '__main__':
    device = 'cpu'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 模型存储路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models', 'matd3_models')
    load_timestamp = "2025-04-15_22-23"  # 替换成你的模型时间戳
    model_timestamp = None if load_timestamp == '' else load_timestamp
    
    # 定义参数
    args = main_parameters()
    args.render_mode = "rgb_array"  # 修改为rgb_array以便捕获帧
    args.episode_num = 5  # 可以根据需要调整评估的回合数

    # 创建环境
    print("Using Env's name", args.env_name)
    # 判断是否使用固定种子
    if args.seed is None:
        print("使用随机种子 (不固定)")
    else:
        print(f"使用固定种子: {args.seed}")
        setup_seed(args.seed)
    
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length, args.render_mode, seed=args.seed)

    # 创建MATD3智能体
    agent = MATD3(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, args.tau, _chkpt_dir=chkpt_dir, _model_timestamp=model_timestamp)
    print("--- Loading models ---")
    agent.load_model()
    print('---- Evaluating and Recording ----')
    
    # 使用修改后的Runner
    runner = RecordingRunner(agent, env, args, device, mode='evaluate')
    frames = runner.evaluate()
    
    # 创建保存目录(如果不存在)
    plot_dir = os.path.join(current_dir, '..', 'plot')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 保存为GIF
    gif_path = os.path.join(plot_dir, f'{args.env_name}_matd3_demo.gif')
    print(f"正在保存GIF到: {gif_path}")
    imageio.mimsave(gif_path, frames, fps=10)
    
    print(f'---- 完成! GIF已保存到 {gif_path} ----')