from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from main_parameters import main_parameters
from utils.runner import RUNNER
from agents.maddpg.MADDPG_agent import MADDPG
import torch
from envs import simple_tag_env
import os
import numpy as np
import imageio  # 需要安装: pip install imageio


def get_env(env_name, ep_len=50, render_mode = "None"):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
        # new_env = simple_tag_env.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    new_env.reset()
    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:",agent_id)
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = [] #[low action,  hign action]
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
        # episode循环
        for episode in range(self.par.episode_num):
            step = 0  # 每回合step重置
            print(f"评估第 {episode + 1} 回合")
            # 初始化环境 返回初始状态
            obs, _ = self.env.reset()  # 重置环境，开始新回合
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
                    print(f"Step {step}, obs: {obs}, action: {action}, reward: {reward}, done: {self.done}")
            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)
                
if __name__ == '__main__':
    device ='cpu'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)
    # 模型存储路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models/maddpg_models/')
    # 加载模型的时间戳
    load_timestamp = "2025-02-19_16-38"
    model_timestamp = None if load_timestamp == '' else load_timestamp
    # 定义参数
    args = main_parameters()
    
    # 设置为rgb_array模式以便捕获帧
    args.render_mode = "rgb_array"  # 修改为rgb_array以便捕获帧
    args.episode_num = 5

    # 创建环境
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length, args.render_mode)
    # print(env, dim_info, action_bound)
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意
    agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _model_timestamp = model_timestamp)
    print("--- Loading models ---")
    agent.load_model()
    print('---- Evaluating and Recording ----')
    
    # 准备录制
    frames = []
    # 使用修改后的Runner
    runner = RecordingRunner(agent, env, args, device, mode='evaluate')
    runner.evaluate()
    
    # 保存为GIF
    gif_path = os.path.join(current_dir, 'plot', f'{args.env_name}_demo.gif')
    print(f"正在保存GIF到: {gif_path}")
    imageio.mimsave(gif_path, frames, fps=10)
    
    print(f'---- 完成! GIF已保存到 {gif_path} ----')