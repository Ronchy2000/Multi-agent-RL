import argparse

def main_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type =str, default = "simple_tag_v3", help = "name of the env",
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument("--render_mode", type=str, default = "None", help = "None | human | rgb_array")
    parser.add_argument("--episode_num", type = int, default = 5000) # 5000
    parser.add_argument("--episode_length", type = int, default = 100) #50
    parser.add_argument('--learn_interval', type=int, default=50,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=500, help='random steps before the agent start to learn') #  2e3
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.7, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=64, help='batch-size of replay buffer')  # 1024
    parser.add_argument('--actor_lr', type=float, default=0.00002, help='learning rate of actor') # .00002
    parser.add_argument('--critic_lr', type=float, default=0.002, help='learning rate of critic') # .002
    # The parameters for the communication network
    # TODO
    parser.add_argument('--visdom', type=bool, default=False, help="Open the visdom")
    parser.add_argument('--size_win', type=int, default=200, help="Open the visdom") # 1000


    args = parser.parse_args()
    return args