import torch
import torch.nn as nn
import torch.functional as F
import os
from datetime import datetime

class MLPNetworkActor(nn.Module):
    def __init__(self,chkpt_name,  chkpt_dir, in_dim, out_dim, action_bound, hidden_dim = 64, non_linear = nn.ReLU()):
        super(MLPNetworkActor, self).__init__()
        # 创建带时间戳到文件夹
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        save_dir = os.path.join(chkpt_dir, timestamp)
        self.chkpt_file = os.path.join(chkpt_dir, chkpt_name)

        # different ,为什么要保持这两个信息？
        self.out_dim = out_dim
        self.action_bound = action_bound

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
        x = self.net(x)
        logi = x
        a_min = self.action_bound[0]
        a_max = self.action_bound[1]
        ''' 这三行为什么要这么处理？ 引入了bias项干嘛'''
        k = torch.tensor( (a_max - a_min) /2 , device=x.device )
        bias = torch.tensor( (a_max + a_min) /2, device=x.device )
        action = k * torch.tanh(x) + bias
        return action, logi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device = 'cpu'):
        self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))