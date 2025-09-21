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

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_env(env_name, ep_len=50, render_mode = "None", seed = None):
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

    new_env.reset(seed)
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



if __name__ == '__main__':
    device ='cpu'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)
    # 模型存储路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, "..", 'models', 'matd3_models')
    load_timestamp = "2025-09-21_19-46" # 替换时间戳
    model_timestamp = None if load_timestamp == '' else load_timestamp
    # 定义参数
    args = main_parameters()
    args.render_mode = "human"
    # args.episode_num = 1

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
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意
    agent = MATD3(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, args.tau, _chkpt_dir = chkpt_dir, _model_timestamp = model_timestamp)
    print("--- Loading models ---")
    agent.load_model()
    print('---- Evaluating ----')
    env.reset(args.seed)
    runner = RUNNER(agent, env, args, device, mode = 'evaluate')
    runner.evaluate() # 使用evaluate方法
    print('---- Done! ----')



