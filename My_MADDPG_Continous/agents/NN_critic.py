import torch
import torch.nn as nn
import torch.functional as F
import os
from datetime import datetime
"""
self.target_critic = CriticNetwork(*, *, 
                                chkpt_dir=chkpt_dir,
                                name=self.agent_name+'_target_critic')
"""
class MLPNetworkCritic(nn.Module):
    def __init__(self, chkpt_name,  chkpt_dir, in_dim, out_dim, hidden_dim = 64, non_linear = nn.ReLU()):
        super(MLPNetworkCritic, self).__init__()
        # 创建时间戳文件夹
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        save_dir = os.path.join(chkpt_dir, timestamp)
        self.chkpt_file = os.path.join(save_dir, chkpt_name)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        '''init patameters of the module'''
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = gain)  #使用了 Xavier 均匀分布初始化（也叫 Glorot 初始化）
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.net(x)
    
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device = 'cpu'):
        self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))
