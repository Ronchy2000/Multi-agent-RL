import torch
import torch.nn as nn
import torch.functional as F
import os
from datetime import datetime

"""
和MADDPG中的actor网络相同.
"""

class MLPNetworkActor_td3(nn.Module):
    def __init__(self,chkpt_name,  chkpt_dir, in_dim, out_dim, action_bound, hidden_dim = 128, non_linear = nn.ReLU()):
        super(MLPNetworkActor_td3, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_name = chkpt_name

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
        # logi = x
        # a_min = self.action_bound[0]
        # a_max = self.action_bound[1]
        # ''' 这三行为什么要这么处理？ 引入了bias项干嘛'''
        # k = torch.tensor( (a_max - a_min) /2 , device=x.device )
        # bias = torch.tensor( (a_max + a_min) /2, device=x.device )
        # action = k * torch.tanh(x) + bias
        # return action, logi
        x = torch.tanh(x)
        return x

    def save_checkpoint(self, is_target=False, timestamp = False):
        # 使用时间戳保存功能
        if timestamp is True:
             # 使用时间戳创建新文件夹
             current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
             save_dir = os.path.join(self.chkpt_dir, current_timestamp)
        else:
            # 直接保存在主目录下，不使用时间戳
            save_dir = self.chkpt_dir
        # 创建保存路径
        self.chkpt_file = os.path.join(save_dir, self.chkpt_name)

        if is_target:
            target_chkpt_name = self.chkpt_file.replace('actor', 'target_actor')
            os.makedirs(os.path.dirname(target_chkpt_name), exist_ok=True)
            torch.save(self.state_dict(), target_chkpt_name)
        else:
            os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
            torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device = 'cpu', is_target = False, timestamp = None): # 默认加载target
        if timestamp and isinstance(timestamp, str):
            # 如果提供了有效的时间戳字符串，从对应文件夹加载
            load_dir = os.path.join(self.chkpt_dir, timestamp)
        else:
            # 否则从主目录加载
            load_dir = self.chkpt_dir

        self.chkpt_file = os.path.join(load_dir, self.chkpt_name)

        if is_target:
                target_chkpt_name = self.chkpt_file.replace('actor', 'target_actor')
                self.load_state_dict(torch.load(target_chkpt_name, map_location=torch.device(device)))
        else:
            self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))