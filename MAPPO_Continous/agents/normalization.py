import numpy as np

class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        # 确保shape是一个元组或整数
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape
            
        self.mean = np.zeros(self.shape, dtype=np.float32)
        self.S = np.zeros(self.shape, dtype=np.float32)
        self.std = np.ones(self.shape, dtype=np.float32)

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        # 确保输入形状与初始化形状匹配
        if x.shape != self.shape:
            if x.size == np.prod(self.shape):
                x = x.reshape(self.shape)
            else:
                raise ValueError(f"输入形状 {x.shape} 与期望形状 {self.shape} 不匹配")
        
        self.n += 1
        if self.n == 1:
            self.mean = x.copy()
            self.std = np.ones_like(x)
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # 将输入转换为numpy数组并确保形状正确
        x = np.asarray(x, dtype=np.float32)
        if update:
            self.running_ms.update(x)
        return (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

class RewardScaling:
    def __init__(self, shape, gamma):
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape, dtype=np.float32)

    def __call__(self, x):
        # 将输入转换为numpy数组
        x = np.asarray(x, dtype=np.float32)
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        return x / (self.running_ms.std + 1e-8)

    def reset(self):
        self.R = np.zeros(self.shape, dtype=np.float32)