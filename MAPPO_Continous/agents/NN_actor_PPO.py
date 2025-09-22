import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *

# Trick 8: orthogonal initialization
'''
函数作用:
权重初始化：将神经网络层的权重矩阵初始化为正交矩阵
偏置初始化：将偏置参数初始化为0
gain参数：控制初始化的缩放因子，影响权重的幅度

正交初始化的优势:
缓解梯度消失/爆炸：正交矩阵的特征值为1，有助于保持梯度在合理范围
加速收敛：良好的初始化可以让网络更快达到最优解
提高训练稳定性：减少训练过程中的数值不稳定问题
'''
def orthogonal_init(layer, gain=1.0): # 正交初始化: 神经网络权重初始化函数
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


# RNN
class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob
    
#######################################################################################################################

# 确保Actor_MLP类正确实现连续动作空间
class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc_mean = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        
        # 初始化log_std为参数，不依赖于状态
        self.log_std = nn.Parameter(torch.zeros(args.action_dim) - 0.5)  # 或初始化为-0.5
        
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc_mean, gain=0.01)
    
    def forward(self, actor_input):
        """
        输入:
            actor_input: 输入特征
            
        输出:
            mean: 高斯分布的均值
            std: 高斯分布的标准差
        """
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        
        # 输出连续动作的均值
        mean = self.fc_mean(x)
        
        # 限制log_std在合理范围内
        log_std = self.log_std.clamp(-20, 2)
        std = log_std.exp()
        
        return mean, std
