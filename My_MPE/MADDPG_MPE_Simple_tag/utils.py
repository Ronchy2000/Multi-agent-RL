# -*- coding: utf-8 -*-
#
# @Time : 2024-10-30 16:05:11
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : utils.py
# @Software: PyCharm
# @Description: None
import torch
import torch.nn
import numpy as np
import torch.nn.functional as F

def trans2onehot(logits, eps=0.01):
    """Transform the output of the actor network to a one-hot vector.

    Args:
        logits:     The output value of the actor network.
        eps:        The epsilon parameter in the epsilon-greedy algorithm used to choose an action.

    Returns: An action in one-hot vector form.

    """
    # Generates a one-hot vector form of the action selected by the actor network.
    best_action = (logits == logits.max(1, keepdim=True)[0]).float()
    # Generate a one-hot vector form of a random action.
    size, num_actions = logits.shape
    random_index = np.random.choice(range(num_actions), size=size)
    random_actions = torch.eye(num_actions)[[random_index]].to(logits.device)
    # Select an action using the epsilon-greedy algorithm.
    random_mask = torch.rand(size, device=logits.device) <= eps
    selected_action = torch.where(random_mask.view(-1, 1), random_actions, best_action)
    return selected_action


def sample_gumbel(shape, eps=1e-20, tensor_type=torch.float32):
    """Sample a Gumbel noise from the Gumbel(0,1) distribution."""
    U = torch.rand(shape, dtype=tensor_type, requires_grad=False)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    return gumbel_noise


def gumbel_softmax_sample(logits, temperature):
    """Sample from the Gumbel-Softmax distribution."""
    gumbel_noise = sample_gumbel(logits.shape, tensor_type=logits.dtype).to(logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """Sample from Gumbel-Softmax distribution and discretize it.

    By returning a one-hot vector of y_hard, but with a gradient of y, we can get both a discrete action interacting with the environment and a correct inverse gradient.
    """
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = trans2onehot(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y

