import numpy as np
import torch

class BUFFER():
    
    def __init__(self,capacity, obs_dim, act_dim, device):
        # 使用连续内存布局
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # 指定dtype
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)  # 使用bool_
        self._index = 0
        self._size = 0
        self.device = device

    def add(self,obs, action, reward, next_obs, done):
        # 确保输入数据类型一致
        self.obs[self._index] = np.asarray(obs, dtype=np.float32)
        self.action[self._index] = np.asarray(action, dtype=np.float32)
        self.reward[self._index] = np.float32(reward)
        self.next_obs[self._index] = np.asarray(next_obs, dtype=np.float32)
        self.done[self._index] = np.float32(done)

        self._index = (self._index +1) % self.capacity
        if self._size < self.capacity:
            self._size += 1


    def sample(self, indices):
        # 一次性批量处理
        batch = (
            self.obs[indices],
            self.action[indices],
            self.reward[indices],
            self.next_obs[indices],
            self.done[indices]
        )
        # 批量转换为tensor并移动到设备
        return tuple(
            torch.as_tensor(data, device=self.device)
            for data in batch
        )

        # obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        # action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        # reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # # reward = (reward - reward.mean()) / (reward.std() + 1e-7) # 暂不使用
        # next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        # done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size
        
        # return obs, action, reward, next_obs, done

    def __len__(self):  #保留方法
        return self._size
        