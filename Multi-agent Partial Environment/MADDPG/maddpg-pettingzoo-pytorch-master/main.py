import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

from MADDPG import MADDPG


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


if __name__ == '__main__':
    #使用命令行传递参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
    #                     choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    # parser.add_argument('--episode_num', type=int, default=30000,
    #                     help='total episode num during training procedure')
    # parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    # parser.add_argument('--learn_interval', type=int, default=100,
    #                     help='steps interval between learning time')
    # parser.add_argument('--random_steps', type=int, default=5e4,
    #                     help='random steps before the agent start to learn')
    # parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    # parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    # parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    # parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    # parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    # args = parser.parse_args()


    #直接在代码中设置参数
    env_name = 'simple_tag_v3'
    episode_num = 30000
    episode_length = 25
    learn_interval = 100  # steps interval between learning time
    random_steps = 50000 # random steps before the agent start to learn
    tau = 0.02 # soft update parameter
    gamma = 0.95 # discounted rate
    buffer_capacity = int(1e6) # capacity of replay buffer
    batch_size = 1024 # batch-size of replay buffer
    actor_lr = 0.01 # learning rate of actor
    critic_lr = 0.01 # learning rate of critic

    # create folder to save result
    # 例如，如果 env_name 是 simple_tag_v2，则 env_dir 将是 ./results/simple_tag_v2
    env_dir = os.path.join('./results', env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)]) #  # 计算目录中已有的文件或文件夹数量
    result_dir = os.path.join(env_dir, f'{total_files + 1}') # 创建一个新的子目录，名称为已有文件数量加1
    # 这段代码将 env_dir 与 total_files + 1 连接起来，生成一个新的子目录路径。
    # 例如，如果 total_files 是 3，则新的子目录路径将是 ./results/simple_adversary_v2/4。
    # 然后，os.makedirs(result_dir) 创建这个新的子目录。
    os.makedirs(result_dir) ## 创建新的子目录

    env, dim_info = get_env(env_name, episode_length)
    maddpg = MADDPG(dim_info, buffer_capacity, batch_size, actor_lr, critic_lr,
                    result_dir)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}
    for episode in range(episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents} # dict 类型
            else:
                action = maddpg.select_action(obs)

            next_obs, reward, done, info = env.step(action)
            # env.render()
            maddpg.add(obs, action, reward, next_obs, done)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= random_steps and step % learn_interval == 0:  # learn every few steps
                maddpg.learn(batch_size, gamma)
                maddpg.update_target(tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

    maddpg.save(episode_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
