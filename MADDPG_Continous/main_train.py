from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from envs import simple_tag_env, custom_agents_dynamics

from main_parameters import main_parameters, setup_seed
from utils.runner import RUNNER

from agents.maddpg.MADDPG_agent import MADDPG
import torch
import os

import time
from datetime import timedelta
from utils.logger import TrainingLogger  # 添加导入

def get_env(env_name, ep_len=25, render_mode ="None", seed=None):
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
    # reset environment with seed when provided to ensure reproducibility
    if seed is not None:
        new_env.reset(seed=seed)
    else:
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
    print("_dim_info:",_dim_info)
    print("action_bound:",action_bound)
    return new_env, _dim_info, action_bound



if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
    #                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print("Using device:",device)
    start_time = time.time() # Record start time
    
    # Model save path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models/maddpg_models/')
    # Define parameters
    args = main_parameters()
    # Set seed if provided
    setup_seed(args.seed)
    # Create environment
    print("Using Env's name",args.env_name)
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length, args.render_mode, seed=args.seed)
    # print(env, dim_info, action_bound)
    # Create MADDPG agent. dim_info: dict with agent names as keys, values are [obs_dim, act_dim]
    agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device)
    # Create runner object
    runner = RUNNER(agent, env, args, device, mode = 'train')
    # Start training
    runner.train()
    print("agent",agent)

    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    # Convert to hours:minutes:seconds format
    training_duration = str(timedelta(seconds=int(training_time)))
    print(f"\n===========Training Complete!===========")
    print(f"Training device: {device}")
    print(f"Training duration: {training_duration}")

    # Save training log using logger
    logger = TrainingLogger()
    current_time = logger.save_training_log(args, device, training_duration, runner)
    print(f"完成时间: {current_time}")

    print("--- saving trained models ---")
    agent.save_model()
    print("--- trained models saved ---")
    


