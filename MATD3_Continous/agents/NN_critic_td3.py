import torch
import torch.nn as nn
import torch.functional as F
import os
from datetime import datetime


''' 创新点1：双截断Q网络'''
class MLPNetworkCritic_td3(nn.Module):
    def __init__(self, chkpt_name,  chkpt_dir, in_dim, out_dim = 1, hidden_dim = 128, non_linear = nn.ReLU()):
        super(MLPNetworkCritic_td3, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_name = chkpt_name
        # Q1网络
        self.net1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)
        # Q2网络
        self.net2 = nn.Sequential(
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
        # 返回两个Q值
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2

    def Q1(self, x):
        # 只使用Q1网络进行评估
        return self.net1(x)
    

    def save_checkpoint(self, is_target = False, timestamp = False):
        if timestamp is True:
            # 使用时间戳创建新文件夹
            current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            save_dir = os.path.join(self.chkpt_dir, current_timestamp)
        else:
            # 直接保存在主目录下
            save_dir = self.chkpt_dir
        
        self.chkpt_file = os.path.join(save_dir, self.chkpt_name)

        if is_target:
            target_chkpt_name = self.chkpt_file.replace('critic', 'target_critic')
            os.makedirs(os.path.dirname(target_chkpt_name), exist_ok=True)
            torch.save(self.state_dict(), target_chkpt_name)
        else:
            os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
            torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device = 'cpu', is_target = False, timestamp = None):
        if timestamp and isinstance(timestamp, str):
            # 如果提供了有效的时间戳字符串，从对应文件夹加载
            load_dir = os.path.join(self.chkpt_dir, timestamp)
        else:
            # 否则从主目录加载
            load_dir = self.chkpt_dir
        
        self.chkpt_file = os.path.join(load_dir, self.chkpt_name)

        if is_target:
            target_chkpt_name = self.chkpt_file.replace('critic', 'target_critic')
            self.load_state_dict(torch.load(target_chkpt_name, map_location=torch.device(device)))
        else:
            self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))
