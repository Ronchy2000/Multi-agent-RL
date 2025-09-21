MODULE_NAME = "log_td3_main" # 使用logger保存训练日志

from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

# 添加项目根目录到Python路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs import simple_tag_env, custom_agents_dynamics

from main_parameters import main_parameters
from agents.MATD3_runner import RUNNER

from agents.MATD3_agent import MATD3
import torch
import random
import numpy as np 

import time
from datetime import datetime, timedelta
from utils.logger import TrainingLogger  # 添加导入


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_env(env_name, ep_len=25, render_mode ="None", seed = None):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_tag_env':
        new_env = simple_tag_env.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    
    # 使用reset时处理None种子
    if seed is not None:
        new_env.reset(seed=seed)  # 指定种子值
    else:
        new_env.reset()  # 不指定种子，使用随机种子

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
    print("_dim_info:",_dim_info)
    print("action_bound:",action_bound)
    return new_env, _dim_info, action_bound


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                            else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print("Using device:",device)
    start_time = time.time() # 记录开始时间
    # 模型保存路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, "..", 'models', 'matd3_models')
    # 定义参数
    args = main_parameters()
    # 创建环境
    print("Using Env's name",args.env_name)

    # 判断是否使用固定种子
    if args.seed is None:
        print("使用随机种子 (不固定)")
    else:
        print(f"使用固定种子: {args.seed}")
        setup_seed(args.seed)
    
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length, args.render_mode, seed = args.seed)
    # print(env, dim_info, action_bound)
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意。
    agent = MATD3(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, args.tau, _chkpt_dir = chkpt_dir, _device = device)
    # 创建运行对象
    runner = RUNNER(agent, env, args, device, mode = 'train')
    
    # 记录训练开始时间
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"训练开始时间: {start_time_str}")
    
    # 开始训练
    runner.train()
    
    # 记录训练结束时间和计算训练用时
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    training_duration = str(timedelta(seconds=int(duration.total_seconds())))
    
    print(f"\n===========训练完成!===========")
    print(f"训练开始时间: {start_time_str}")
    print(f"训练结束时间: {end_time_str}")
    print(f"训练用时: {training_duration}")
    print(f"训练设备: {device}")

    # 使用logger保存训练日志
    logger = TrainingLogger(module_name = MODULE_NAME)
    logger.save_training_log(args, device, start_time_str, end_time_str, training_duration, runner)

    print("--- saving trained models ---")
    agent.save_model(timestamp = True)
    print("--- trained models saved ---")
    


