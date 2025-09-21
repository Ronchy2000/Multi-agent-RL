import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
from NN_critic_PPO import Critic_RNN, Critic_MLP
from NN_actor_PPO import Actor_RNN, Actor_MLP
import os

class MAPPO_MPE:
    def __init__(self, args):
        self.args = args

        # 从环境获取的参数
        self.N = len(args.agents)  # 智能体数量
        
        # 确定每个智能体的类型和观测维度
        self.agent_types = {}
        self.obs_dims = {}
        for agent_id in args.agents:
            # 确定智能体类型
            if agent_id.startswith('adversary_'):
                self.agent_types[agent_id] = 'adversary'
            else:
                self.agent_types[agent_id] = 'agent'
            
            # 获取智能体观测维度
            if hasattr(args, 'dim_info') and agent_id in args.dim_info:
                self.obs_dims[agent_id] = args.dim_info[agent_id][0]
            else:
                # 默认处理
                self.obs_dims[agent_id] = 12 if self.agent_types[agent_id] == 'adversary' else 10
              
        # 创建多个actor网络，每种观测维度一个
        self.actors = {}
        self.actor_input_dims = {}
        
        # 获取所有不同的观测维度
        unique_obs_dims = set(self.obs_dims.values())
        for obs_dim in unique_obs_dims:
            actor_input_dim = obs_dim
            if self.args.add_agent_id:
                actor_input_dim += self.N
                
            self.actor_input_dims[obs_dim] = actor_input_dim
            
            if self.args.use_rnn:
                self.actors[obs_dim] = Actor_RNN(args, actor_input_dim)
            else:
                self.actors[obs_dim] = Actor_MLP(args, actor_input_dim)
        
        # 创建critic网络
        self.critic_input_dim = self.args.state_dim
        if self.args.add_agent_id:
            print("------添加智能体ID------")
            self.critic_input_dim += self.N
            
        if self.args.use_rnn:
            print("------使用RNN网络------")
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.critic = Critic_MLP(args, self.critic_input_dim)
        
        # 设置优化器
        self.ac_parameters = []
        for actor in self.actors.values():
            self.ac_parameters.extend(list(actor.parameters()))
        self.ac_parameters.extend(list(self.critic.parameters()))
        
        if self.args.set_adam_eps:
            print("------设置Adam epsilon------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.args.actor_lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.args.actor_lr)
    
    @staticmethod
    def tanh_action_sample(dist):
        """从正态分布中采样动作并应用tanh压缩"""
        raw_action = dist.rsample()  # 使用rsample()支持重参数化
        action = torch.tanh(raw_action)
        # 修正log_prob：加上tanh的导数项
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)
        # 加上映射到[0,1]的雅可比行列式（链式法则）
        # log_prob -= np.log(2) * action.shape[-1]
        return action, log_prob

    def choose_action(self, obs_n, evaluate, env_agents=None):
        """
        选择动作
        
        参数:
            obs_n: 可以是字典(dict)格式 {agent_id: obs} 或 numpy.ndarray格式
            evaluate: 是否为评估模式
            env_agents: 环境中智能体的顺序列表，当obs_n为字典时需要
        """
        with torch.no_grad():
            # 处理输入格式
            if isinstance(obs_n, dict):
                if env_agents is None:
                    raise ValueError("当obs_n为字典格式时，需要提供env_agents参数")
                
                # 为每个智能体生成动作
                actions = []
                log_probs = []
                
                for i, agent_id in enumerate(env_agents):
                    if agent_id not in obs_n:
                        raise ValueError(f"在obs_n中找不到智能体 {agent_id}")
                    
                    # 获取当前智能体的观测
                    agent_obs = obs_n[agent_id]
                    agent_obs = np.asarray(agent_obs, dtype=np.float32)
                    agent_obs_tensor = torch.from_numpy(agent_obs).unsqueeze(0)  # shape: (1, obs_dim)
                    
                    # 确定当前智能体的观测维度
                    obs_dim = agent_obs_tensor.shape[-1]
                    
                    # 根据观测维度选择对应的actor网络
                    if obs_dim in self.actors:
                        actor = self.actors[obs_dim]
                    else:
                        # 如果没有对应维度的actor，使用第一个可用的
                        actor = list(self.actors.values())[0]
                    
                    # 构建actor输入
                    actor_inputs = [agent_obs_tensor]
                    if self.args.add_agent_id:
                        # 为当前智能体创建one-hot编码
                        agent_id_tensor = torch.zeros(1, self.N, dtype=torch.float32)
                        agent_id_tensor[0, i] = 1.0
                        actor_inputs.append(agent_id_tensor)

                    actor_inputs = torch.cat(actor_inputs, dim=-1)  # (1, actor_input_dim)

                    # 连续动作
                    mean, std = actor(actor_inputs)
                    dist = torch.distributions.Normal(mean, std)
                    
                    if evaluate:
                        action = torch.tanh(mean)  # 保持在[-1,1]
                        log_prob = None
                    else:
                        # 采样并进行tanh-squash, 同时返回修正后的log_prob
                        raw_action = dist.rsample()
                        action = torch.tanh(raw_action)
                        # log_prob修正（tanh的雅可比）
                        log_prob = dist.log_prob(raw_action).sum(-1)
                        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)
                    
                    actions.append(action.cpu().numpy()[0])
                    if log_prob is not None:
                        log_probs.append(log_prob.cpu().numpy()[0])
                
                a_np = np.array(actions)
                lp_np = np.array(log_probs) if log_probs else None
                
                return a_np, lp_np
                
            else:
                # 处理numpy数组输入（向后兼容）
                # 假设所有智能体有相同的观测维度
                obs_n = np.asarray(obs_n, dtype=np.float32)
                obs_tensor = torch.from_numpy(obs_n)  # shape: (N, obs_dim)
                
                # 确定观测维度并选择actor
                obs_dim = obs_tensor.shape[-1]
                if obs_dim in self.actors:
                    actor = self.actors[obs_dim]
                else:
                    actor = list(self.actors.values())[0]

                actor_inputs = [obs_tensor]
                if self.args.add_agent_id:
                    actor_inputs.append(torch.eye(self.N, dtype=torch.float32))

                actor_inputs = torch.cat(actor_inputs, dim=-1)  # (N, actor_input_dim)

                # 连续动作
                mean, std = actor(actor_inputs)
                dist = torch.distributions.Normal(mean, std)
                if evaluate:
                    a_n = torch.tanh(mean)  # 保持在[-1,1]
                    a_logprob_n = None
                else:
                    # 采样并进行tanh-squash, 同时返回修正后的log_prob
                    raw_action = dist.rsample()
                    a_n = torch.tanh(raw_action)
                    # log_prob修正（tanh的雅可比）
                    log_prob = dist.log_prob(raw_action).sum(-1)
                    log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)
                    a_logprob_n = log_prob

                a_np = a_n.cpu().numpy()
                lp_np = None if a_logprob_n is None else a_logprob_n.cpu().numpy()
                return a_np, lp_np

    def get_value(self, s):
        """获取状态值函数
        s: 全局状态
        """
        with torch.no_grad():
            critic_inputs = []
            # 每个智能体都有相同的全局状态
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
            critic_inputs.append(s)
            
            if self.args.add_agent_id:
                critic_inputs.append(torch.eye(self.N))
                
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
            v_n = self.critic(critic_inputs)
            
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        """训练网络
        replay_buffer: 经验回放缓冲区
        total_steps: 当前总步数
        """
        batch = replay_buffer.get_training_data()  # 获取训练数据
        
        # 使用GAE计算优势函数
        adv = []
        gae = 0
        with torch.no_grad():  # adv和td_target不需要梯度
            # 计算时序差分误差
            deltas = batch['r_n'] + self.args.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]
            
            # 反向计算GAE
            for t in reversed(range(self.args.episode_length)):
                gae = deltas[:, t] + self.args.gamma * self.args.lamda * gae
                adv.insert(0, gae)
                
            adv = torch.stack(adv, dim=1)  # adv.shape=(batch_size, episode_length, N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape=(batch_size, episode_length, N)
            
            if self.args.use_adv_norm:  # Trick 1: 优势函数归一化
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
                
        # 获取critic输入
        critic_inputs = self.get_critic_inputs(batch)
        
        # 进行K轮优化
        for _ in range(self.args.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.args.batch_size)), self.args.mini_batch_size, False):
                # 处理每种观测维度的智能体
                actor_loss_all = 0
                dist_entropy_all = 0
                
                for obs_dim, agent_indices in batch['agent_indices_by_dim'].items():
                    # 获取当前维度的观测数据
                    actor_inputs = batch['obs_n_by_dim'][obs_dim][index]
                    actor = self.actors[obs_dim]
                    
                    # 获取当前策略的动作分布
                    mean_now, std_now = actor(actor_inputs)
                    dist_now = torch.distributions.Normal(mean_now, std_now)
                    
                    # 提取相应智能体的动作数据
                    a_n = batch['a_n'][index, :, agent_indices, :]
                    a_logprob_n = batch['a_logprob_n'][index, :, agent_indices]
                    agent_adv = adv[index, :, agent_indices]
                    
                    # 计算动作对数概率（包含tanh变换）
                    eps = 1e-6
                    a_n_raw = torch.atanh(torch.clamp(a_n, -1 + eps, 1 - eps))
                    log_prob = dist_now.log_prob(a_n_raw).sum(-1)
                    log_prob -= (2 * (np.log(2) - a_n_raw - F.softplus(-2 * a_n_raw))).sum(-1)
                    
                    # 计算重要性权重
                    ratios = torch.exp(log_prob - a_logprob_n.detach())
                    
                    # PPO裁剪目标
                    surr1 = ratios * agent_adv
                    surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * agent_adv
                    
                    # 策略熵
                    dist_entropy = dist_now.entropy().sum(-1)
                    
                    # 计算actor损失
                    actor_loss = -torch.min(surr1, surr2) - self.args.entropy_coef * dist_entropy
                    
                    # 累加所有智能体的损失
                    actor_loss_all += actor_loss.mean()
                    dist_entropy_all += dist_entropy.mean()
                
                # 计算critic损失
                values_now = self.critic(critic_inputs[index]).squeeze(-1)
                
                # 值函数裁剪
                if self.args.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.args.epsilon, self.args.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                
                # 总损失
                ac_loss = actor_loss_all + critic_loss.mean()
                
                # 优化
                self.ac_optimizer.zero_grad()
                ac_loss.backward()
                
                if self.args.use_grad_clip:  # Trick 7: 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                    
                self.ac_optimizer.step()
        
        if self.args.use_lr_decay:
            self.args.actor_lr_decay(total_steps)

    def get_critic_inputs(self, batch):
        """获取critic网络输入"""
        critic_inputs = []
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        
        if self.args.add_agent_id:
            # agent_id_one_hot.shape=(batch_size, episode_length, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.args.batch_size, self.args.episode_length, 1, 1)
            critic_inputs.append(agent_id_one_hot)
            
        return torch.cat([x for x in critic_inputs], dim=-1)

    def lr_decay(self, total_steps):
        """学习率衰减
        total_steps: 当前总步数
        """
        lr_now = self.args.actor_lr * (1 - total_steps / self.args.episode_num)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        """获取网络输入
        batch: 经验回放缓冲区中的批量数据
        """
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        
        if self.args.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.args.batch_size, self.args.episode_length, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)
            
        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        
        return actor_inputs, critic_inputs

    def save_model(self, timestamp=False):
        """保存模型
        timestamp: 是否在文件名中添加时间戳
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, '../models/mappo_models')
        os.makedirs(model_dir, exist_ok=True)
        
        # 使用时间戳或默认文件名
        if timestamp:
            from datetime import datetime
            timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
            prefix = f"mappo_{timestamp_str}_"
        else:
            prefix = "mappo_"
        
        # 保存每个actor网络
        for obs_dim, actor in self.actors.items():
            model_path = os.path.join(model_dir, f"{prefix}actor_{obs_dim}.pth")
            torch.save(actor.state_dict(), model_path)
        
        # 保存critic网络
        critic_path = os.path.join(model_dir, f"{prefix}critic.pth")
        torch.save(self.critic.state_dict(), critic_path)
        
        print(f"模型已保存到 {model_dir}")
        
    def load_model(self, model_timestamp=None):
        """加载模型
        model_timestamp: 模型时间戳
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, '../models/mappo_models')
        
        # 使用时间戳或默认文件名
        if model_timestamp:
            prefix = f"mappo_{model_timestamp}_"
        else:
            prefix = "mappo_"
        
        # 加载每个actor网络
        for obs_dim, actor in self.actors.items():
            model_path = os.path.join(model_dir, f"{prefix}actor_{obs_dim}.pth")
            if os.path.exists(model_path):
                actor.load_state_dict(torch.load(model_path))
                print(f"已加载actor模型: {model_path}")
            else:
                print(f"未找到actor模型: {model_path}")
        
        # 加载critic网络
        critic_path = os.path.join(model_dir, f"{prefix}critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"已加载critic模型: {critic_path}")
        else:
            print(f"未找到critic模型: {critic_path}")