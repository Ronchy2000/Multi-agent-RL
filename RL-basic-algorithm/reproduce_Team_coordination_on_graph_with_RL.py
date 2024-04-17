import numpy as np

# 假设我们有4个节点和2个代理
V = 4
N = 2

# 创建代理位置的one-hot向量
P = np.zeros((V, N))
# 假设代理1在节点0，代理2在节点1
P[0, 0] = 1
P[1, 1] = 1

# 创建邻接矩阵
ADJ = np.full((V, V), np.inf)  # 初始时所有边的权重都是无穷大
# 假设节点0和节点1之间有边，权重为1
ADJ[0, 1] = 1
ADJ[1, 0] = 1


# 创建支持张量
SUP = np.full((V, V, V), np.inf)  # 初始时所有的支持成本都是无穷大
# 假设代理1在节点0，可以支持节点1到节点2的转移，降低成本为0.5
SUP[0, 1, 2] = 0.5
SUP[0, 2, 1] = 0.5

# 将所有的部分合并为一个状态
S = (P.flatten(), ADJ, SUP)

print("Agent positions (one-hot):")
print(P)
print("Adjacency matrix:")
print(ADJ)
print("Support tensor:")
print(SUP)

print("State:",S)