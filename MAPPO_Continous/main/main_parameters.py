import argparse

def main_parameters():
    parser = argparse.ArgumentParser("MAPPO Parameters")
    ############################################ 选择环境 ############################################
    parser.add_argument("--seed", type=int, default=-1, help='随机种子 (使用-1表示不使用固定种子)')
    parser.add_argument("--use_variable_seeds", type=bool, default=False, help="使用可变随机种子")
    
    parser.add_argument("--env_name", type=str, default="simple_tag_v3", help="环境名称",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env']) 
    parser.add_argument("--render_mode", type=str, default="None", help="渲染模式: None | human | rgb_array")
    
    # MAPPO训练参数
    parser.add_argument("--episode_num", type=int, default=int(3e6), help="训练轮数") # max_train_steps
    parser.add_argument("--episode_length", type=int, default=25, help="每轮最大步数")
    parser.add_argument("--evaluate_episode_num", type=int, default=100, help="评估轮数")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="每隔多少步评估一次策略")
    parser.add_argument("--evaluate_times", type=int, default=3, help="每次评估重复次数")
    
    # 学习参数
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小(回合数)')
    parser.add_argument('--mini_batch_size', type=int, default=8, help='小批量大小(回合数)')
    parser.add_argument('--actor_lr', type=float, default=5e-4, help='Actor学习率')
    parser.add_argument('--critic_lr', type=float, default=5e-4, help='Critic学习率')
    parser.add_argument('--lamda', type=float, default=0.95, help='GAE参数')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PPO裁剪参数')
    parser.add_argument('--K_epochs', type=int, default=15, help='每批数据训练轮数')
    
    # MAPPO网络参数
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="RNN隐藏层维度")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="MLP隐藏层维度")
    
    # MAPPO技巧参数
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:优势归一化")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:奖励归一化")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:奖励缩放")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5:策略熵系数")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:学习率衰减")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7:梯度裁剪")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8:正交初始化")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9:设置Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="是否使用ReLU激活函数,False则使用tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="是否使用RNN网络")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="是否添加智能体ID")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="是否使用值函数裁剪")
    
    # 经验回放参数
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='经验回放缓冲区容量')
    parser.add_argument('--learn_interval', type=int, default=10, help='学习间隔步数')
    parser.add_argument('--random_steps', type=int, default=500, help='初始随机探索步数')
    
    # 其他算法参数
    parser.add_argument('--tau', type=float, default=0.01, help='软更新参数')
    parser.add_argument('--comm_lr', type=float, default=0.00001, help='通信网络学习率')
    parser.add_argument('--message_dim', type=int, default=3, help='通信消息维度')
    parser.add_argument('--best_score', type=int, default=-20, help='最佳分数初始值')
    
    # 可视化参数
    parser.add_argument('--visdom', action="store_true", help="是否使用visdom可视化")
    parser.add_argument('--size_win', type=int, default=200, help="平滑窗口大小")
    
    # 训练设备
    parser.add_argument("--device", type=str, default='cpu', help="训练设备，cpu或cuda")
    
    # 模型保存与加载
    parser.add_argument("--save_interval", type=int, default=100000, help="模型保存间隔步数")
    parser.add_argument("--model_dir", type=str, default="../models/mappo_models", help="模型保存目录")
    parser.add_argument("--load_model", type=bool, default=False, help="是否加载预训练模型")
    parser.add_argument("--model_timestamp", type=str, default="", help="要加载的模型时间戳")

    args = parser.parse_args()
    
    # 参数后处理
    if args.seed == -1:
        args.seed = None
    
    # 确保episode_limit与episode_length一致
    args.episode_limit = args.episode_length
    
    return args