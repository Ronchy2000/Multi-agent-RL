# -*- coding: utf-8 -*-
#
# @Time : 2024-10-30 16:05:55
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : NN_Module.py
# @Software: PyCharm
# @Description: None

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """SimpleNet is a 3-layer simple neural network.

    It's used to approximate the policy functions and the value functions.
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        """Initial a simple network for an actor or a critic.

        Args:
            input_dim: The input dimension of the network.
            output_dim: The output dimension of the network.
            hidden_dim: The hidden layer dimension of the network.
        """
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)