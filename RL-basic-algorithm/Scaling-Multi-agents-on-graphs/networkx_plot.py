
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


'''
一直邻接矩阵，利用networkx画图
'''
# 假设你的邻接矩阵是 adj_matrix
adj_matrix = np.matrix([[0, 10, 0], [10, 0, 10], [0, 10, 0]])

# 使用邻接矩阵创建一个图
G = nx.from_numpy_array(adj_matrix)

# 绘制图
nx.draw(G, with_labels=True)
plt.show()
