# -*- coding: utf-8 -*-
#
# @Time : 2024-10-30 11:14:07
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : ReplayBuffer.py
# @Software: PyCharm
# @Description: None
# Import libraries
import random
from collections import namedtuple, deque

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
# from pettingzoo.mpe import simple_adversary_v3
from pettingzoo.mpe import simple_tag_v3
from psutil import virtual_memory


"""Transition is a namedtuple used to store a transition.

namedtuple：允许你创建一个类似于元组的对象，但你可以通过名称来访问字段，而不是通过位置。这样使你的代码更具可读性。
e.g.
t = Transition(state=1, action=2, reward=3, next_state=4, done=False)
print(t.state)  # 输出 1

The structure of Transition looks like this:
    (state, action, reward, next_state, done)
"""

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """ReplayBuffer is a class used to achieve experience replay.
    deque：双端队列，支持从两端快速添加和删除元素，常用作队列和栈的数据结构
    It's a buffer composed of a deque with a certain capacity. When the deque is full, it will automatically remove the oldest transition in the buffer.

    Attributes:
        _storage: The buffer to store the transitions.

    Examples:
        A replay buffer structure looks like below:
        [
            (state_1, action_1, reward_1, next_state_1, done_1),
            (state_2, action_2, reward_2, next_state_2, done_2),
            ...
            (state_n, action_n, reward_n, next_state_n, done_n),
        ]
        Each tuple is a transition.
    """

    def __init__(self, capacity=100_000):
        """Initial a replay buffer with capacity.

        Args:
            capacity: Max length of the buffer.
        """
        self._storage = deque([], maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        Args:
            state:          The state of the agents.
            action:         The action of the agents.
            reward:         The reward of the agents.
            next_state:     The next state of the agents.
            done:           The termination of the agents.

        Returns: None

        """
        transition = Transition(state, action, reward, next_state, done)
        self._storage.append(transition)

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: The number of transitions that we want to sample from the buffer.

        Returns: A batch of transitions.

        Example:
            Assuming that batch_size=3, we'll randomly sample 3 transitions from the buffer:
                state_batch = (state_1, state_2, state_3)
                action_batch = (action_1, action_2, action_3)
                reward_batch = (reward_1, reward_2, reward_3)
                next_state_batch = (next_state_1, next_state_2, next_state_3)
                done_batch = (done_1, done_2, done_3)
        """
        transitions = random.sample(self._storage, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

        """
        zip(*transitions) 是一个强大的Python技巧。
        zip 函数将多个可迭代对象（如列表或元组）“压缩”到一起，返回由这些对象中相应元素组成的元组的迭代器。
        在这种情况下，你有一个包含多个 Transition 命名元组的列表 transitions。zip(*transitions) 会将这个列表“解压缩”，
        然后创建几个新的元组，每个元组包含每个 Transition 对应属性的值。
        transitions = [
        Transition(state=1, action=2, reward=3, next_state=4, done=False),
        Transition(state=5, action=6, reward=7, next_state=8, done=True)
        ]
        zip(*transitions) 将产生如下结果：
        [(1, 5), (2, 6), (3, 7), (4, 8), (False, True)]
        因此， 
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions) 
        将每个属性的值分别分组到各自的批处理中：

        state_batch 将是 (1, 5)
        action_batch 将是 (2, 6)
        reward_batch 将是 (3, 7)
        next_state_batch 将是 (4, 8)
        done_batch 将是 (False, True)
        """
    def __len__(self):
        """Return the length of the buffer."""
        return len(self._storage)
