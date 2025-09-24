import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import *
import numpy as np
import os

# 网络初始化函数
def net_init(m, gain=None, use_relu=True):
    '''网络初始化'''
    use_orthogonal = True
    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_]
    activate_function = ['tanh', 'relu', 'leaky_relu']
    gain = gain if gain is not None else nn.init.calculate_gain(activate_function[use_relu])
    
    init_method[use_orthogonal](m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)

# Trick 8: orthogonal initialization (保持原有接口)
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128, trick=None):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.trick = trick if trick is not None else {}
        
        # 使用 orthogonal_init
        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.mean_layer, gain=0.01)

    def forward(self, x):
        if self.trick.get('feature_norm', False):
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l1(x))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])

        mean = torch.tanh(self.mean_layer(x))
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean, std

class Actor_discrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128, trick=None):
        super(Actor_discrete, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

        self.trick = trick if trick is not None else {}
        
        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3, gain=0.01)

    def forward(self, obs):
        if self.trick.get('feature_norm', False):
            obs = F.layer_norm(obs, obs.size()[1:])
        x = F.relu(self.l1(obs))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])
        a_prob = torch.softmax(self.l3(x), dim=1)
        return a_prob

class Critic(nn.Module):
    def __init__(self, dim_info, hidden_1=128, hidden_2=128, trick=None):
        super(Critic, self).__init__()
        # 使用原始观察维度计算全局状态维度
        self.dim_info = dim_info
        self.agent_obs_dims = [val[0] for val in dim_info.values()]
        global_obs_dim = sum(self.agent_obs_dims)
        
        self.l1 = nn.Linear(global_obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        
        self.trick = trick if trick is not None else {}
        
        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3)

    def forward(self, s):
        # s应该是全局状态张量 (batch_size, global_state_dim)
        if isinstance(s, (list, tuple)) and len(s) == 1:
            s = s[0]
        elif isinstance(s, dict):
            # 如果输入是字典，按照dim_info的顺序拼接
            s_parts = []
            for agent_id, (obs_dim, _) in self.dim_info.items():
                if agent_id in s:
                    s_parts.append(s[agent_id])
                else:
                    # 为缺失的智能体填充零
                    batch_size = next(iter(s.values())).shape[0]
                    s_parts.append(torch.zeros(batch_size, obs_dim, device=next(iter(s.values())).device))
            s = torch.cat(s_parts, dim=1)
            
        if self.trick.get('feature_norm', False):
            s = F.layer_norm(s, s.size()[1:])
        
        q = F.relu(self.l1(s))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = self.l3(q)

        return q

class Agent:
    def __init__(self, obs_dim, action_dim, dim_info, actor_lr, critic_lr, is_continue, device, trick):
        if is_continue:
            self.actor = Actor(obs_dim, action_dim, trick=trick).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=trick).to(device)
        self.critic = Critic(dim_info, trick=trick).to(device)

        if trick.get('adam_eps', False):
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def tanh_action_sample(dist):
    raw_action = dist.rsample()
    action = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action).sum(-1)
    log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)
    return action, log_prob

class HAPPO_MPE:
    def __init__(self, args):
        self.args = args
        self.env = None
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = getattr(args, 'rnn_hidden_dim', 64)

        # 异构智能体相关
        self.all_agents = []
        self.agent_index = {}
        self.agents = {}  # 新的异构智能体字典
        self.dim_info = {}  # 存储每个智能体的维度信息
        
        # 训练参数
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip
        self.device = args.device
        
        # 异构构建标志
        self._hetero_built = False
        
        # 配置tricks
        self.trick = {
            'adv_norm': self.use_adv_norm,
            'orthogonal_init': getattr(args, 'use_orthogonal_init', True),
            'adam_eps': self.set_adam_eps,
            'lr_decay': self.use_lr_decay,
            'ValueClip': self.use_value_clip,
            'feature_norm': False,
            'LayerNorm': False,
            'huber_loss': False,
        }

    def _build_hetero_if_needed(self):
        if self._hetero_built:
            return
        if not self.all_agents or len(self.all_agents) == 0:
            return

        # 获取每个智能体的观察维度
        if self.env is not None:
            obs_dims = self.env.get_obs_dims() if hasattr(self.env, 'get_obs_dims') else {}
        else:
            obs_dims = {}

        # 构建dim_info
        for agent_id in self.all_agents:
            if agent_id in obs_dims:
                obs_dim = obs_dims[agent_id]
            else:
                obs_dim = self.obs_dim  # 使用默认值
            self.dim_info[agent_id] = [obs_dim, self.action_dim]

        # 创建异构智能体
        for agent_id in self.all_agents:
            obs_dim = self.dim_info[agent_id][0]
            if self.add_agent_id:
                obs_dim += self.N
                
            self.agents[agent_id] = Agent(
                obs_dim=obs_dim,
                action_dim=self.action_dim,
                dim_info=self.dim_info,
                actor_lr=self.lr,
                critic_lr=self.lr,
                is_continue=True,  # 连续动作空间
                device=self.device,
                trick=self.trick
            )

        # 设置agent索引
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.all_agents)}
        self._hetero_built = True

    def reset_rnn_hidden(self):
        """保持原有接口"""
        pass  # 新架构不使用RNN，保持接口兼容性

    def choose_action(self, obs_dict, evaluate):
        self._build_hetero_if_needed()
        with torch.no_grad():
            active_agents = list(obs_dict.keys())
            actions_dict = {}
            logprobs_dict = {}
            
            for agent_id in active_agents:
                obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                if self.add_agent_id:
                    idx = self.agent_index.get(agent_id, 0)
                    one_hot = torch.eye(self.N)[idx].unsqueeze(0).to(self.device)
                    actor_input = torch.cat([obs, one_hot], dim=-1)
                else:
                    actor_input = obs
                    
                mean, std = self.agents[agent_id].actor(actor_input)
                dist = Normal(mean, std)
                
                if evaluate:
                    a = torch.tanh(mean)
                    actions_dict[agent_id] = a.squeeze(0).cpu().numpy()
                else:
                    a, logp = tanh_action_sample(dist)
                    actions_dict[agent_id] = a.squeeze(0).cpu().numpy()
                    logprobs_dict[agent_id] = logp.squeeze(0).item()
                    
            if evaluate:
                return actions_dict, None
            return actions_dict, logprobs_dict

    def get_value(self, s):
        self._build_hetero_if_needed()
        with torch.no_grad():
            values = []
            # s是全局状态，维度为 state_dim
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            for agent_id in self.all_agents:
                # 每个智能体的Critic都接收相同的全局状态
                v = self.agents[agent_id].critic(s_tensor)
                values.append(v.squeeze(-1))
                
            v_n = torch.stack(values, dim=0)
            return v_n.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps):
        self._build_hetero_if_needed()
        batch = replay_buffer.get_training_data()

        # 计算GAE优势
        adv = []
        gae = 0
        with torch.no_grad():
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v_n'][:, :-1]
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # HAPPO: 随机打乱智能体更新顺序
        shuffled_agents = np.random.permutation(self.all_agents).tolist()
        
        # 初始化保护因子
        factor = torch.ones((self.batch_size, self.episode_limit, 1), device=adv.device)
        
        # 预处理动作
        eps = 1e-6
        a_n_raw_full = torch.atanh(torch.clamp(batch['a_n'], -1 + eps, 1 - eps))
        
        # 按随机顺序更新每个智能体
        for agent_id in shuffled_agents:
            idx = self.agent_index[agent_id]
            
            # 获取该智能体的真实观察维度
            agent_obs_dim = self.dim_info[agent_id][0]
            
            # 从填充的观察中提取该智能体的实际观察
            obs_agent_padded = batch['obs_n'][:, :, idx, :]  # (B, T, max_obs_dim)
            obs_agent = obs_agent_padded[:, :, :agent_obs_dim]  # (B, T, agent_obs_dim)
            a_agent_raw = a_n_raw_full[:, :, idx, :]
            
            # 构建输入
            if self.add_agent_id:
                one_hot = torch.eye(self.N, device=adv.device)[idx].view(1, 1, -1).repeat(self.batch_size, self.episode_limit, 1)
                actor_input = torch.cat([obs_agent, one_hot], dim=-1)
            else:
                actor_input = obs_agent
                
            # 计算更新前的log概率
            with torch.no_grad():
                mean_old, std_old = self.agents[agent_id].actor(actor_input)
                dist_old = Normal(mean_old, std_old)
                log_prob_old = dist_old.log_prob(a_agent_raw).sum(-1, keepdim=True)
                log_prob_old -= (2 * (np.log(2) - a_agent_raw - F.softplus(-2 * a_agent_raw))).sum(-1, keepdim=True)
            
            # 多次优化该智能体
            for _ in range(self.K_epochs):
                for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                    # 当前批次数据
                    factor_batch = factor[index]
                    actor_input_batch = actor_input[index]
                    a_agent_raw_batch = a_agent_raw[index]
                    adv_batch = adv[index, :, idx].unsqueeze(-1)
                    v_target_batch = v_target[index, :, idx].unsqueeze(-1)
                    
                    # 计算当前策略
                    mean, std = self.agents[agent_id].actor(actor_input_batch)
                    dist = Normal(mean, std)
                    
                    # 计算新策略的log概率
                    log_prob = dist.log_prob(a_agent_raw_batch).sum(-1, keepdim=True)
                    log_prob -= (2 * (np.log(2) - a_agent_raw_batch - F.softplus(-2 * a_agent_raw_batch))).sum(-1, keepdim=True)
                    
                    # 计算策略比率
                    log_prob_old_batch = log_prob_old[index]
                    ratios = torch.exp(log_prob - log_prob_old_batch)
                    
                    # HAPPO: 应用保护因子修正优势
                    surr1 = ratios * adv_batch * factor_batch
                    surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * adv_batch * factor_batch
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算熵损失
                    entropy = dist.entropy().sum(-1, keepdim=True).mean()
                    actor_loss -= self.entropy_coef * entropy
                    
                    # 更新actor
                    self.agents[agent_id].update_actor(actor_loss)
                    
                    # 计算critic值 - 直接使用全局状态
                    s_global_batch = batch['s'][index]  # (mini_batch_size, episode_limit, state_dim)
                    
                    # 将批次数据重塑为 (batch_size * episode_limit, state_dim)
                    s_global_reshaped = s_global_batch.view(-1, s_global_batch.size(-1))
                    v_now = self.agents[agent_id].critic(s_global_reshaped)
                    # 重塑回 (batch_size, episode_limit, 1)
                    v_now = v_now.view(len(index), self.episode_limit, 1)
                    
                    # 计算critic损失
                    if self.use_value_clip:
                        v_old = batch['v_n'][index, :-1, idx].unsqueeze(-1).detach()
                        v_clipped = v_old + torch.clamp(v_now - v_old, -self.epsilon, self.epsilon)
                        critic_loss1 = (v_now - v_target_batch)**2
                        critic_loss2 = (v_clipped - v_target_batch)**2
                        critic_loss = torch.max(critic_loss1, critic_loss2).mean()
                    else:
                        critic_loss = ((v_now - v_target_batch)**2).mean()
                    
                    # 更新critic
                    self.agents[agent_id].update_critic(critic_loss)
            
            # HAPPO: 更新保护因子
            with torch.no_grad():
                mean_new, std_new = self.agents[agent_id].actor(actor_input)
                dist_new = Normal(mean_new, std_new)
                log_prob_new = dist_new.log_prob(a_agent_raw).sum(-1, keepdim=True)
                log_prob_new -= (2 * (np.log(2) - a_agent_raw - F.softplus(-2 * a_agent_raw))).sum(-1, keepdim=True)
                
                factor = factor * torch.exp(log_prob_new - log_prob_old).detach()
        
        # 学习率衰减
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for agent_id in self.all_agents:
            for p in self.agents[agent_id].actor_optimizer.param_groups:
                p['lr'] = lr_now
            for p in self.agents[agent_id].critic_optimizer.param_groups:
                p['lr'] = lr_now

    def save_model(self, env_name, number, seed, total_steps, time_stamp):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        timestamp_dir = os.path.join(models_dir, time_stamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        
        actor_path = os.path.join(timestamp_dir, f"HAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        
        self._build_hetero_if_needed()
        # 添加调试信息
        print(f"保存模型到: {actor_path}")
        print(f"智能体列表: {self.all_agents}")
        print(f"构建的智能体: {list(self.agents.keys())}")
        save_obj = {
            'format': 'hetero_per_agent_actor_v2',
            'agents': self.all_agents,
            'actor_state_dict_by_agent': {aid: self.agents[aid].actor.state_dict() for aid in self.all_agents}
        }
        torch.save(save_obj, actor_path)

    def load_model(self, env_name, number, seed, step, timestamp=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        
        if timestamp:
            timestamp_dir = os.path.join(models_dir, timestamp)
        else:
            timestamp_dir = models_dir
            
        actor_path = os.path.join(timestamp_dir, f"HAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
        
        data = torch.load(actor_path)
        self._build_hetero_if_needed()
        
        if isinstance(data, dict) and 'actor_state_dict_by_agent' in data:
            for aid, state in data['actor_state_dict_by_agent'].items():
                if aid in self.agents:
                    self.agents[aid].actor.load_state_dict(state)
            print("加载异构模型:", actor_path)
        else:
            for aid in self.agents:
                self.agents[aid].actor.load_state_dict(data)
            print("加载共享模型并复制到所有智能体:", actor_path)

    # 为了兼容性，保留旧的接口
    @property
    def actor_by_agent(self):
        return {aid: agent.actor for aid, agent in self.agents.items()}
    
    @property
    def critic_by_agent(self):
        return {aid: agent.critic for aid, agent in self.agents.items()}