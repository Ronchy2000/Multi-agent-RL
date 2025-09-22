import numpy as np
import torch

class ReplayBuffer:
    """
    MAPPO的经验回放缓冲区，支持异构智能体(不同的观测维度)和连续动作空间
    """
    def __init__(self, args):
        self.args = args
        self.N = len(args.agents)  # 智能体数量
        self.obs_dims = {}  # 每个智能体的观测维度
        
        # 获取每个智能体的观测维度
        for agent_id in args.agents:
            if hasattr(args, 'dim_info') and agent_id in args.dim_info:
                self.obs_dims[agent_id] = args.dim_info[agent_id][0]
            else:
                # 默认处理 - 根据agent类型设置不同的观测维度
                self.obs_dims[agent_id] = 12 if agent_id.startswith('adversary_') else 10
        
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        
    def reset_buffer(self):
        """重置缓冲区，使用对象数组存储不同维度的观测"""
        # 使用对象数组存储不同维度的观测
        self.buffer = {
            'obs_n': np.zeros([self.args.batch_size, self.args.episode_num, self.N], dtype=object),
            's': np.zeros([self.args.batch_size, self.args.episode_num, self.args.state_dim], dtype=np.float32),
            'v_n': np.zeros([self.args.batch_size, self.args.episode_num + 1, self.N], dtype=np.float32),
            'a_n': np.zeros([self.args.batch_size, self.args.episode_num, self.N, self.args.action_dim], dtype=np.float32),
            'a_logprob_n': np.zeros([self.args.batch_size, self.args.episode_num, self.N], dtype=np.float32),
            'r_n': np.zeros([self.args.batch_size, self.args.episode_num, self.N], dtype=np.float32),
            'done_n': np.zeros([self.args.batch_size, self.args.episode_num, self.N], dtype=np.float32)
        }
        
        # 初始化观测数组
        for idx in range(self.args.batch_size):
            for step in range(self.args.episode_num):
                for i, agent_id in enumerate(self.args.agents):
                    self.buffer['obs_n'][idx, step, i] = np.zeros(self.obs_dims[agent_id], dtype=np.float32)
        
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        """存储一个转移
        
        参数:
            episode_step: 当前episode中的步数
            obs_n: 观测，字典格式 {agent_id: obs} 或 numpy数组格式 (N, obs_dim)
            s: 全局状态
            v_n: 值函数估计
            a_n: 连续动作，numpy数组 (N, action_dim)
            a_logprob_n: 动作对数概率
            r_n: 奖励
            done_n: 完成状态
        """
        idx = self.episode_num
        
        # 将字典转换为列表
        if isinstance(obs_n, dict):
            obs_list = []
            for agent_id in self.args.agents:
                if agent_id in obs_n:
                    obs_list.append(obs_n[agent_id])
            obs_n = obs_list
        
        # 存储数据
        for i, obs in enumerate(obs_n):
            self.buffer['obs_n'][idx, episode_step, i] = np.array(obs, dtype=np.float32)
        
        self.buffer['s'][idx, episode_step] = s
        self.buffer['v_n'][idx, episode_step] = v_n
        self.buffer['a_n'][idx, episode_step] = a_n
        self.buffer['a_logprob_n'][idx, episode_step] = a_logprob_n
        self.buffer['r_n'][idx, episode_step] = r_n
        self.buffer['done_n'][idx, episode_step] = done_n
    
    def store_last_value(self, episode_step, v_n):
        """存储最后一步的值函数"""
        idx = self.episode_num
        self.buffer['v_n'][idx, episode_step] = v_n
        self.episode_num = (self.episode_num + 1) % self.args.batch_size  # 更新episode计数器
        
    def get_training_data(self):
        """获取训练数据"""
        batch = {}
        
        # 处理观测数据 - 为每种观测维度创建单独的张量
        obs_by_dim = {}
        agent_indices_by_dim = {}
        
        # 按观测维度分组智能体
        for i, agent_id in enumerate(self.args.agents):
            obs_dim = self.obs_dims[agent_id]
            if obs_dim not in obs_by_dim:
                obs_by_dim[obs_dim] = []
                agent_indices_by_dim[obs_dim] = []
            agent_indices_by_dim[obs_dim].append(i)
        
        # 创建batch['obs_n_by_dim']，包含按维度分组的观测
        batch['obs_n_by_dim'] = {}
        batch['agent_indices_by_dim'] = agent_indices_by_dim
        
        for obs_dim, indices in agent_indices_by_dim.items():
            # 创建这个维度的观测张量
            obs_tensor = np.zeros((self.args.batch_size, self.args.episode_num, len(indices), obs_dim), dtype=np.float32)
            
            # 填充数据
            for b_idx in range(self.args.batch_size):
                for step in range(self.args.episode_num):
                    for i, agent_idx in enumerate(indices):
                        obs_tensor[b_idx, step, i] = self.buffer['obs_n'][b_idx, step, agent_idx]
            
            batch['obs_n_by_dim'][obs_dim] = torch.tensor(obs_tensor, dtype=torch.float32)
        
        # 转换其他数据为张量
        batch['s'] = torch.tensor(self.buffer['s'], dtype=torch.float32)
        batch['v_n'] = torch.tensor(self.buffer['v_n'], dtype=torch.float32)
        batch['a_n'] = torch.tensor(self.buffer['a_n'], dtype=torch.float32)
        batch['a_logprob_n'] = torch.tensor(self.buffer['a_logprob_n'], dtype=torch.float32)
        batch['r_n'] = torch.tensor(self.buffer['r_n'], dtype=torch.float32)
        batch['done_n'] = torch.tensor(self.buffer['done_n'], dtype=torch.float32)
        
        return batch