from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from main_parameters import main_parameters

# 修改导入路径
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.MAPPO_runner import Runner_MAPPO_MPE
from agents.MAPPO_agent import MAPPO_MPE
import torch
import random
import numpy as np
from envs import simple_tag_env

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
    """创建环境并获取每个智能体在该环境中的观测和动作维度"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode=render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_tag_env':
        new_env = simple_tag_env.parallel_env(render_mode=render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    
    # 重置环境
    if seed is not None:
        new_env.reset(seed=seed)
    else:
        new_env.reset()
        
    # 获取环境信息
    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:", agent_id)
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = []  # [low action, high action]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)
    
    # 将环境信息添加到args
    args = main_parameters()
    args.agents = new_env.agents
    args.obs_dim = _dim_info[list(new_env.agents)[0]][0]  # 假设所有智能体观测维度相同
    args.action_dim = _dim_info[list(new_env.agents)[0]][1]  # 假设所有智能体动作维度相同
    args.state_dim = sum(_dim_info[agent_id][0] for agent_id in new_env.agents)  # 状态维度是所有智能体观测维度之和
    
    return new_env, _dim_info, action_bound, args

if __name__ == '__main__':
    device = 'cpu'
    print("使用设备:", device)
    
    # 模型存储路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models', 'mappo_models')
    
    # 加载指定时间戳的模型
    load_timestamp = "2025-07-02_16-10"  # 请根据实际保存的模型时间戳修改
    model_timestamp = None if load_timestamp == '' else load_timestamp
    
    # 定义参数
    args = main_parameters()
    args.render_mode = "human"  # 设置为人类可视化模式
    # 创建环境
    print("使用环境:", args.env_name)
    
    # 判断是否使用固定种子
    if args.seed is None:
        print("使用随机种子 (不固定)")
    else:
        print(f"使用固定种子: {args.seed}")
        setup_seed(args.seed)
    
    env, dim_info, action_bound, args = get_env(args.env_name, args.episode_length, args.render_mode, seed=args.seed)
    
    # 创建MAPPO智能体
    agent = MAPPO_MPE(args)
    
    print("--- 加载模型 ---")
    agent.load_model(model_timestamp=model_timestamp)
    
    print('---- 开始评估 ----')
    runner = Runner_MAPPO_MPE(agent, env, args, device, mode='evaluate')
    capture_rate, avg_capture_steps = runner.evaluate()
    
    print(f'---- 评估完成! ----')
    print(f'捕获成功率: {capture_rate:.4f}')
    print(f'平均捕获步数: {avg_capture_steps:.4f}')