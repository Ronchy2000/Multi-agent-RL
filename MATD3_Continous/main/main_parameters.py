import argparse
import torch
import numpy as np
import random


def setup_seed(seed):
    """Set random seeds for torch, numpy and python.random.
    If seed is None, do not set deterministic behavior.
    """
    if seed is None:
        print("No fixed seed set, using random seed")
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Fixed seed set to: {seed}")


def main_parameters():
    parser = argparse.ArgumentParser("MADDPG legacy")
    ############################################ Environment Selection ############################################
    parser.add_argument("--seed", type=int, default=None, help='Random seed (None for random seed)')
    parser.add_argument("--use_variable_seeds", type=bool, default=False, help="Use variable random seeds")
    
    parser.add_argument("--env_name", type=str, default="simple_tag_v3", help="name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env']) 
    parser.add_argument("--render_mode", type=str, default="None", help="None | human | rgb_array")
    parser.add_argument("--episode_num", type=int, default=1500, help="训练轮数")
    parser.add_argument("--episode_length", type=int, default=100, help="每轮最大步数") # 训练时设置为100，测试时嫌麻烦可以设置为50
    parser.add_argument("--evaluate_episode_num", type=int, default=100, help="评估轮数")
    parser.add_argument('--learn_interval', type=int, default=10,
                        help='学习间隔步数')
    
    parser.add_argument('--random_steps', type=int, default=200, help='初始随机探索步数')
    parser.add_argument('--tau', type=float, default=0.01, help='软更新参数')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='经验回放缓冲区容量')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--actor_lr', type=float, default=0.00001, help='Actor学习率')
    parser.add_argument('--critic_lr', type=float, default=0.0001, help='Critic学习率')
    parser.add_argument('--comm_lr', type=float, default=0.00001, help='Comm学习率')
    # 通信网络参数
    parser.add_argument('--message_dim', type=int, default=3, help='通信消息维度')
    
    parser.add_argument('--best_score', type=int, default= -20, help='最佳分数_初始值')

    # 可视化参数
    parser.add_argument('--visdom', action="store_true", help="是否使用visdom可视化")
    parser.add_argument('--size_win', type=int, default=200, help="平滑窗口大小")
    
    # 训练设备
    parser.add_argument("--device", type=str, default='cpu', help="训练设备，默认自动选择cpu")

    args = parser.parse_args()
        
    return args