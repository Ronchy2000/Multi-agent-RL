import numpy as np
import matplotlib.pyplot as plt

class Importance_sampling:
    def __init__(self,p0_probability, p1_probability):
        self.seed = 42
        self.p0_probability = p0_probability
        self.p1_probability = p1_probability
        self.p1_values = [1, -1]
        self.p1_samples = np.array([])
    def sampling(self,sampling_size):
        # generate samples
        np.random.seed(self.seed)
        self.p1_samples = np.random.choice(self.p1_values, size=sampling_size, p=self.p1_probability)

        print("p1_samples::", self.p1_samples.shape) # 采样结果 向量。
    def calculate(self):
        if not self.p1_samples.size: #if p1_samples is empty, raise error.   self.sample返回的不是一个bool值
            raise ValueError("Please generate p1_samples first")
        # 计算累积和
        cumulative_sum = np.cumsum(self.p1_samples)
        # 计算累积平均值
        p1_samples_average = cumulative_sum / np.arange(1, len(self.p1_samples) + 1)

        p_xi1 = np.where(self.p1_samples == -1, self.p1_probability[1], self.p1_probability[0])
        p_xi0 = np.where(self.p1_samples == -1, self.p0_probability[1], self.p0_probability[0])
        # 计算
        importance_p0_samples = (p_xi0/p_xi1) * self.p1_samples  #Core，importance sampling 的体现


        cumulative_sum = np.cumsum(importance_p0_samples)
        cumulative_importance_p0_average = cumulative_sum / np.arange(1, len(importance_p0_samples) + 1)

        return p1_samples_average, cumulative_importance_p0_average
    def render(self,average_result, importance_sampling_result):
        plt.figure(figsize=(10, 6)) # set size of figure


        x1 = np.arange(len(self.p1_samples[:200]))
        y1 = self.p1_samples[:200]
        # plt.xlim(0, x.shape[0])  # adaptive is fine.
        plt.ylim(-2, 2) # set x,y range
        plt.plot(x1, y1, 'ro', markerfacecolor='none', label='p0_samples')

        y0 = average_result[:200]
        plt.plot(x1, y0, 'b.', label='average')

        y2 = importance_sampling_result[:200]
        plt.plot(x1, y2, 'g-', label='importance sampling')


        plt.xlabel('Sample index')
        # plt.ylabel()
        plt.legend() #图中带标签
        plt.show()



if __name__ == '__main__':
    p0_probability = [0.5, 0.5]
    p1_probability = [0.8, 0.2]
    importance_sampling = Importance_sampling(p0_probability, p1_probability) #实例化

    importance_sampling.sampling(200)
    average_result, importance_sampling_result = importance_sampling.calculate()

    importance_sampling.render(average_result, importance_sampling_result)

    print("Done!")
