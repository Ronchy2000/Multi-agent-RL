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

    def EG_constuction(self):
        # self.environment_graph = nx.Graph()
        self.environment_graph.add_nodes_from(self.nodes)
        self.environment_graph.add_edges_from(self.edges)

    def EG_render(self):
        nx.draw(self.environment_graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=18)


if __name__ == "__main__":
    nodes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
             16: 16, 17: 17,
             18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29}

    edges = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 3): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1,
             (1, 6): 1, (2, 8): 1, (3, 6): 1, (3, 8): 1, (4, 6): 1, (6, 7): 1, (4, 7): 1, (4, 9): 1,
             (8, 4): 1, (8, 9): 1, (5, 7): 1, (5, 9): 1,
             (1, 10): 1, (6, 10): 1, (6, 11): 1, (10, 11): 1, (7, 11): 1, (7, 12): 1, (11, 12): 1, (6, 17): 1,
             (12, 13): 1, (12, 14): 1, (12, 15): 1, (13, 15): 1, (13, 19): 1, (5, 17): 1, (14, 18): 1, (15, 19): 1,
             (16, 19): 1, (18, 19): 1, (16, 17): 1,
             (7, 13): 1, (5, 16): 1, (14, 15): 1, (20, 22): 1, (20, 29): 1, (22, 9): 1, (22, 23): 1,
             (23, 8): 1, (23, 24): 1, (24, 2): 1, (24, 25): 1, (25, 26): 1, (26, 27): 1, (26, 23): 1,
             (27, 28): 1, (27, 21): 1, (29, 21): 1, (29, 28): 1, (22, 28): 1, (17, 20): 1, (23, 28): 1, (22, 8): 1}

    EG = EnvironmentGraph(nodes, edges)
    EG.EG_constuction()
    EG.EG_render()
    plt.show()