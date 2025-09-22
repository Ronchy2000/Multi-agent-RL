import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from MAPPO import MAPPO_MPE

from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3


def get_env(env_name, ep_len=25, render_mode ="None", seed = None , N = 1):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, N = N, render_mode="None", continuous_actions=True)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    # if env_name == 'simple_tag_env':
    #     new_env = simple_tag_env.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    
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

class Runner_MAPPO_MPE:
    def __init__(self, args, number, seed, env_name = 'simple_spread_v3'):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env, _, _ = get_env(env_name = 'simple_spread_v3', N = self.number) # Continous action space
        print("self.env.agents",self.env.agents)
        self.args.agents = self.env.agents
        self.args.N = len(self.env.agents)
        print("self.env.observation_space:", self.env.observation_space)
        self.args.obs_dim_n = [self.env.observation_space(i).shape[0] for i in self.args.agents]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space(i).shape[0] for i in self.args.agents]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.agent_n.env = self.env  # 设置环境引用
        self.replay_buffer = ReplayBuffer(self.args)

        # 保存所有智能体ID列表，用于后续引用
        self.agent_n.all_agents = [agent_id for agent_id in self.env.agents]

        # Create a tensorboard
        # self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        # self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        # np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_dict, _ = self.env.reset()  # 接收字典格式的观察 # obs_n.shape=(N，obs_dim)
        # 重置done标志
        done_dict = {agent_id: False for agent_id in self.agent_n.all_agents}
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            # 现在 choose_action 返回动作字典
            actions_dict, logprobs_dict = self.agent_n.choose_action(obs_dict, evaluate=evaluate) # Get actions and the corresponding log probabilities of N agents
            # 返回 a_n: shape (N, action_dim)，是连续动作
            # a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            
            # 将观察转换为数组，确保使用所有原始智能体ID（即使某些可能已经不在env.agents中）
            obs_n = []
            for agent_id in self.agent_n.all_agents:
                if agent_id in obs_dict:
                    obs_n.append(obs_dict[agent_id])
                else:
                    # 对于已经完成的智能体，使用零填充
                    obs_n.append(np.zeros(self.args.obs_dim))

            # 为了计算状态值，我们仍然需要将观察转换为列表
            obs_n = np.array(obs_n)
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            # obs_next_n, r_n, done_n, _, _ = self.env.step(a_n)
            next_obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions_dict)
            ## episode_reward += r_n[0]
            # episode_reward += r_n
            # 计算总奖励
            step_reward = sum(rewards_dict.values())
            episode_reward += step_reward
            
            # 更新done标志
            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done_dict[agent_id] = True
            # 确定是否结束
            done = all(done_dict.values()) or len(self.env.agents) == 0
            done_n = np.array([done] * self.args.N)  # 对所有智能体使用相同的done标志

            if not evaluate:
                a_n = np.zeros((self.args.N, self.args.act_dim))
                r_n = np.zeros(self.args.N)
                # 为活动智能体填充实际值
                for i, agent_id in enumerate(self.agent_n.all_agents):
                    if agent_id in actions_dict:
                        a_n[i] = actions_dict[agent_id]
                    if agent_id in rewards_dict:
                        r_n[i] = rewards_dict[agent_id]
                        
                # 如果logprobs_dict不是None，则转换为列表
                # 处理logprobs
                if logprobs_dict is not None:
                    a_logprob_n = np.zeros(self.args.N)
                    for i, agent_id in enumerate(self.agent_n.all_agents):
                        if agent_id in logprobs_dict:
                            a_logprob_n[i] = logprobs_dict[agent_id]
                else:
                    a_logprob_n = None
                # 应用奖励归一化
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_dict = next_obs_dict
            if done:
                break

        if not evaluate:
            # 一个episode结束，在最后一步存储v_n
            obs_n = np.zeros((self.args.N, self.args.obs_dim))
            for i, agent_id in enumerate(self.agent_n.all_agents):
                if agent_id in obs_dict:
                    obs_n[i] = obs_dict[agent_id]
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument("--act_dim", type=float, default=5, help="Act_dimension") # 修改连续时添加

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, number=1, seed=0)
    runner.run()