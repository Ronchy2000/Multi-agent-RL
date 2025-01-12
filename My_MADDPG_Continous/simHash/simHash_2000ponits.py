import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 生成2000个随机点，范围在(-1, 1)
num_points = 2000
D = 2  # 每个点的维度，2维点
points = np.random.uniform(-1, 1, (num_points, D))

# SimHash参数
A = None  # 随机矩阵A稍后动态设置

# SimHash函数定义
def simhash(points, A):
    # 计算点与矩阵A的内积
    projections = np.dot(points, A.T)  # 点集与A矩阵内积
    # 对每个维度的投影结果取符号，生成哈希码
    hash_codes = np.sign(projections)  # 得到的hash值是-1, 0, 1
    return hash_codes

# 将哈希值分组
def group_by_hash(points, hash_codes):
    # 将哈希值转换为tuple（便于区分分组）
    hash_tuples = [tuple(code) for code in hash_codes]
    # 分组
    unique_hashes = list(set(hash_tuples))  # 所有唯一的哈希值
    grouped_points = {hash_value: [] for hash_value in unique_hashes}
    for point, hash_value in zip(points, hash_tuples):
        grouped_points[hash_value].append(point)
    return grouped_points

# 绘制分组结果
def plot_simhash_grouping(points, grouped_points, k):
    plt.figure(figsize=(6, 6))
    colormap = matplotlib.colormaps['tab20']  # 颜色映射
    colors = colormap(np.linspace(0, 1, len(grouped_points)))  # 分配颜色

    # 绘制不同组的点
    for color, (hash_value, group_points) in zip(colors, grouped_points.items()):
        group_points = np.array(group_points)
        plt.scatter(group_points[:, 0], group_points[:, 1], color=color, label=str(hash_value), alpha=0.7)
    
    plt.title(f'SimHash Grouping with k={k}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # plt.show()

# 不同k值的效果
k_values = [8, 16, 32]
for k in k_values:
    A = np.random.randn(k, D)  # 随机生成A矩阵，k行D列
    hash_codes = simhash(points, A)  # 计算哈希值
    grouped_points = group_by_hash(points, hash_codes)  # 分组
    plot_simhash_grouping(points, grouped_points, k)
    # plt.pause(1)  # 暂停1秒，方便观察
# plt.show() #统一在最后显示图像