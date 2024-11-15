from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from envs import simple_tag_env

from main_parameters import main_parameters
from utils.runner import RUNNER

from agents.MADDPG_agent import MADDPG
import torch
def get_env(env_name, ep_len=25, render_mode ="None"):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    if env_name == 'simple_tag_v3':
        # new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
        new_env = simple_tag_env.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
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



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # 模型保存路径
    chkpt_dir='models/maddpg_models/'
    # 定义参数
    args = main_parameters()
    # 创建环境
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length, args.render_mode)
    # print(env, dim_info, action_bound)
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意
    agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device)
    # 创建运行对象
    runner = RUNNER(agent, env, args, device)
    # 开始训练
    runner.train()
    print("agent",agent)
    print("--- saving trained models ---")
    agent.save_model()
    print("--- trained models saved ---")
    


