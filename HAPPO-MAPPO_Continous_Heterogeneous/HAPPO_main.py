import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from HAPPO import HAPPO_MPE

from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
import os
from datetime import datetime
import csv

def get_env(env_name, num_good, num_adversaries, ep_len=25, render_mode="None", seed=None):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode=render_mode, continuous_actions=True)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode=render_mode, num_good=num_good, num_adversaries=num_adversaries, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    
    # 使用reset时处理None种子
    if seed is not None:
        new_env.reset(seed=seed)
    else:
        new_env.reset()

    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:", agent_id)
        _dim_info[agent_id] = []
        action_bound[agent_id] = []
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)
    print("_dim_info:", _dim_info)
    print("action_bound:", action_bound)
    return new_env, _dim_info, action_bound

class Runner_MAPPO_MPE:
    def __init__(self, args, num_good, num_adversaries, seed, env_name):
        self.args = args
        self.env_name = env_name
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.seed = seed
        self.number = getattr(args, 'number', 1)  # 添加默认值
        
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 创建训练开始时间戳
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        print(f"训练开始时间戳: {self.timestamp}")
        
        # Create env
        self.env, self.dim_info, _ = get_env(env_name=env_name, num_good=self.num_good, num_adversaries=self.num_adversaries, ep_len=self.args.episode_limit)
        print("self.env.agents", self.env.agents)

        # 根据智能体ID区分追捕者和逃跑者
        self.adversary_agents = []
        self.good_agents = []
        
        for agent_id in self.env.agents:
            if agent_id.startswith('adversary_'):
                self.adversary_agents.append(agent_id)
            else:
                self.good_agents.append(agent_id)
        
        print(f"追捕者智能体: {self.adversary_agents}")
        print(f"逃跑者智能体: {self.good_agents}")
        
        # 设置环境参数
        self.args.agents = self.env.agents
        self.args.N = len(self.env.agents)
        self.args.obs_dim_n = [self.env.observation_space(i).shape[0] for i in self.args.agents]
        self.args.action_dim_n = [self.env.action_space(i).shape[0] for i in self.args.agents]
        
        # 为了处理异构观察，我们使用最大观察维度
        self.max_obs_dim = max(self.args.obs_dim_n)
        self.args.obs_dim = self.max_obs_dim  # 使用最大观察维度
        self.args.action_dim = self.args.action_dim_n[0]
        
        # 计算全局状态维度（所有智能体观察的总和）
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("max_obs_dim={}".format(self.max_obs_dim))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create agent with heterogeneous support
        self.agent_n = HAPPO_MPE(self.args)
        self.agent_n.env = self.env
        self.agent_n.all_agents = [agent_id for agent_id in self.env.agents]
        self.agent_n.dim_info = self.dim_info  # 设置维度信息
        
        # 为环境添加get_obs_dims方法
        def get_obs_dims():
            return {agent_id: self.env.observation_space(agent_id).shape[0] for agent_id in self.env.agents}
        self.env.get_obs_dims = get_obs_dims
        
        self.replay_buffer = ReplayBuffer(self.args)

        self.evaluate_rewards = []
        self.total_steps = 0
        
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def _pad_obs_to_max_dim(self, obs_dict):
        """将不同维度的观察填充到最大维度 - 仅用于replay buffer存储"""
        obs_list = []
        for agent_id in self.agent_n.all_agents:
            if agent_id in obs_dict:
                obs = obs_dict[agent_id]
                # 填充到最大维度
                if len(obs) < self.max_obs_dim:
                    padded_obs = np.zeros(self.max_obs_dim)
                    padded_obs[:len(obs)] = obs
                    obs_list.append(padded_obs)
                else:
                    obs_list.append(obs)
            else:
                # 对于已经完成的智能体，使用零填充
                obs_list.append(np.zeros(self.max_obs_dim))
        return np.array(obs_list)

    def _get_global_state(self, obs_dict):
        """获取全局状态（保持原始维度）- 用于Critic网络"""
        state_parts = []
        for agent_id in self.agent_n.all_agents:
            if agent_id in obs_dict:
                state_parts.append(obs_dict[agent_id])
            else:
                # 对于已完成的智能体，使用原始观察维度的零填充
                agent_obs_dim = self.dim_info[agent_id][0]
                state_parts.append(np.zeros(agent_obs_dim))
        return np.concatenate(state_parts)
    
    def _obs_dict_to_individual_tensors(self, obs_dict):
        """将观察字典转换为各智能体的独立观察（保持原始维度）"""
        obs_tensors = {}
        for agent_id in self.agent_n.all_agents:
            if agent_id in obs_dict:
                obs_tensors[agent_id] = obs_dict[agent_id]
            else:
                # 为缺失的智能体创建零观察
                agent_obs_dim = self.dim_info[agent_id][0]
                obs_tensors[agent_id] = np.zeros(agent_obs_dim)
        return obs_tensors

    def run(self):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.save_rewards_to_csv()
        self.env.close()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        
        # Save the model
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps, self.timestamp)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_dict, _ = self.env.reset()
        done_dict = {agent_id: False for agent_id in self.agent_n.all_agents}
        
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.reset_rnn_hidden()
            
        for episode_step in range(self.args.episode_limit):
            # 选择动作
            actions_dict, logprobs_dict = self.agent_n.choose_action(obs_dict, evaluate=evaluate)
            
            # 将观察转换为填充后的数组（所有智能体使用相同维度）
            obs_n = self._pad_obs_to_max_dim(obs_dict)
            
            # 获取全局状态（保持原始维度拼接）
            s = self._get_global_state(obs_dict)
            v_n = self.agent_n.get_value(s)
            
            # 环境步进
            next_obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions_dict)
            
            # 计算总奖励
            step_reward = sum(rewards_dict.values())
            episode_reward += step_reward
            
            # 更新done标志
            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done_dict[agent_id] = True
                    
            done = all(done_dict.values()) or len(self.env.agents) == 0
            done_n = np.array([done] * self.args.N)

            if not evaluate:
                a_n = np.zeros((self.args.N, self.args.act_dim))
                r_n = np.zeros(self.args.N)
                
                # 填充实际值
                for i, agent_id in enumerate(self.agent_n.all_agents):
                    if agent_id in actions_dict:
                        a_n[i] = actions_dict[agent_id]
                    if agent_id in rewards_dict:
                        r_n[i] = rewards_dict[agent_id]
                        
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

                # 存储转换 - 注意这里obs_n已经是填充后的数组
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_dict = next_obs_dict
            if done:
                break

        if not evaluate:
            # 存储最后的值
            obs_n = self._pad_obs_to_max_dim(obs_dict)
            s = self._get_global_state(obs_dict)
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1

    def save_rewards_to_csv(self):
        """保存评估奖励数据到CSV文件"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, f"happo_rewards_{self.env_name}_n{self.number}_s{self.seed}_{self.timestamp}.csv")
        
        steps = [i * self.args.evaluate_freq for i in range(len(self.evaluate_rewards))]
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Steps', 'Reward'])
            for step, reward in zip(steps, self.evaluate_rewards):
                writer.writerow([step, reward])
        print(f"评估奖励数据已保存到 {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for HAPPO in MPE environment")
    parser.add_argument("--device", type=str, default='cpu', help="training device")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=1000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO update epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip")
    parser.add_argument("--act_dim", type=float, default=5, help="Act_dimension")
    parser.add_argument("--number", type=int, default=1, help="Experiment number")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, num_good=1, num_adversaries=3, seed=23, env_name="simple_tag_v3")
    runner.run()