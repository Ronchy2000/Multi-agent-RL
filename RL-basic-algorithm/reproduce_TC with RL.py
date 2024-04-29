
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EnvironmentGraph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.graph = nx.Graph()
        self.create_graph()
        self.agents_positions = {i: None for i in range(num_nodes)}  # 初始化智能体位置

    def create_graph(self):
        # 随机生成图结构
        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.rand() < 0.5:
                    edges.append((i, j))
        self.graph.add_edges_from(edges)

        # 随机分配每个智能体到图中的一个节点
        for agent in self.agents_positions:
            self.agents_positions[agent] = np.random.choice(list(self.graph.nodes()))

    def visualize(self, pos, agents_positions):
        nx.draw(self.graph, pos, with_labels=True)  # 绘制图
        # 绘制智能体
        for agent, position in agents_positions.items():
            plt.gca().text(pos[position][0], pos[position][1], f'A{agent+1}', fontsize=12)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.gca().set_aspect('equal', adjustable='box')

    def update(self, frame, pos, agents_positions):
        for agent in agents_positions:
            position = agents_positions[agent]
            possible_actions = list(self.graph.neighbors(position))
            if possible_actions:  # 确保智能体不是孤立的
                new_position = np.random.choice(possible_actions)
                agents_positions[agent] = new_position
        return agents_positions

# 初始化环境
env = EnvironmentGraph(num_nodes=5)

# 创建图形并显示
fig, ax = plt.subplots()
pos = nx.spring_layout(env.graph)  # 为图设置布局

# 初始化智能体的位置
agents_positions = {agent: pos for agent, pos in env.agents_positions.items()}
env.visualize(pos, agents_positions)

# 创建动画
ani = FuncAnimation(fig, lambda frame, pos, agents_positions: env.update(frame, pos, agents_positions),
                    fargs=(pos, agents_positions), blit=True, interval=500)

plt.show()