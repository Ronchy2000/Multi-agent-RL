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
    parser = argparse.ArgumentParser()
    ############################################ Environment Selection ############################################
    parser.add_argument("--env_name", type =str, default = "simple_tag_v3", help = "name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env']) 
    parser.add_argument("--render_mode", type=str, default = "None", help = "None | human | rgb_array")
    parser.add_argument("--episode_num", type = int, default = 5) # default: 5000, test: 5
    parser.add_argument("--episode_length", type = int, default = 500) #50
    parser.add_argument('--learn_interval', type=int, default=10,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=500, help='random steps before the agent start to learn') #  2e3
    parser.add_argument('--tau', type=float, default=0.001, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=128, help='batch-size of replay buffer')  
    parser.add_argument('--actor_lr', type=float, default=0.0002, help='learning rate of actor') # .00002
    parser.add_argument('--critic_lr', type=float, default=0.002, help='learning rate of critic') # .002
    # The parameters for the communication network
    # TODO
    parser.add_argument('--visdom', type=bool, default=False, help="Open the visdom")
    parser.add_argument('--size_win', type=int, default=200, help="Open the visdom") # 1000
    # Seed for reproducibility. Use None for random seed.
    parser.add_argument('--seed', type=int, default=20, help='Random seed (None for random)')


    args = parser.parse_args()
    return args