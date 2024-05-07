import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os


'''
以节点列表和边列表的方式创建图
'''
class EnvironmentGraph():
    def __init__(self,nodes, edges, risky_edge = 0, start_nodes = [], target_nodes = []):
        self.nodes = nodes
        self.edges = edges
        self.risky_edge = risky_edge
        self.start_nodes = start_nodes
        self.target_nodes = target_nodes
        self.environment_graph = nx.Graph()
        self.adj_matrix = []

        self.EG_constuction()
        self.EG_render()

    def EG_constuction(self):
        # self.environment_graph = nx.Graph()
        self.environment_graph.add_nodes_from(self.nodes)
        self.environment_graph.add_edges_from(self.edges)
        self.adj_matrix = nx.adjacency_matrix(self.environment_graph).todense()  # 返回图的邻接矩阵
        print(self.adj_matrix)

    def EG_render(self):
        nx.draw(self.environment_graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=18)


if __name__ == "__main__":
    #生成的无向图
    nodes = list(range(10))
    edges = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 3): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1,
             (1, 6): 1, (2, 8): 1, (3, 6): 1, (3, 8): 1, (4, 6): 1, (6, 7): 1, (4, 7): 1, (4, 9): 1,
             (8, 4): 1, (8, 9): 1, (5, 7): 1, (5, 9): 1}

    EG = EnvironmentGraph(nodes, edges)
    print("是否为对称阵:{}".format((np.array(EG.adj_matrix).T == np.array(EG.adj_matrix)).all()))  #https://blog.csdn.net/tintinetmilou/article/details/78555486  判断两矩阵相等，此处判断：对称阵
    plt.show()


