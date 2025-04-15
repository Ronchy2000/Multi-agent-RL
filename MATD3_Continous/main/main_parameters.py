import argparse

def main_parameters():
    parser = argparse.ArgumentParser("MADDPG legacy")
    ############################################ 选择环境 ############################################
    parser.add_argument("--seed", type=int, default=-1, help='随机种子 (使用-1表示不使用固定种子)')
    parser.add_argument("--use_variable_seeds", type=bool, default=False, help="使用可变随机种子")
    
    parser.add_argument("--env_name", type=str, default="simple_tag_v3", help="name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env']) 
    parser.add_argument("--render_mode", type=str, default="None", help="None | human | rgb_array")
    parser.add_argument("--episode_num", type=int, default=3, help="训练轮数")
    parser.add_argument("--episode_length", type=int, default=100, help="每轮最大步数")
    parser.add_argument("--evaluate_episode_num", type=int, default=100, help="评估轮数")
    parser.add_argument('--learn_interval', type=int, default=10,
                        help='学习间隔步数')
    
    parser.add_argument('--random_steps', type=int, default=500, help='初始随机探索步数')
    parser.add_argument('--tau', type=float, default=0.01, help='软更新参数')
    parser.add_argument('--gamma', type=float, default=0.9, help='折扣因子')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='经验回放缓冲区容量')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='Actor学习率')
    parser.add_argument('--critic_lr', type=float, default=0.003, help='Critic学习率')
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
    
    # 如果seed为-1，则设置为None
    if args.seed == -1:
        args.seed = None
        
    return args