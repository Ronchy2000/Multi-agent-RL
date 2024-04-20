# import numpy as np
#
# # 假设我们有4个节点和2个代理
# V = 4
# N = 2
#
# # 创建代理位置的one-hot向量
# P = np.zeros((V, N))
# # 假设代理1在节点0，代理2在节点1
# P[0, 0] = 1
# P[1, 1] = 1
#
# # 创建邻接矩阵
# ADJ = np.full((V, V), np.inf)  # 初始时所有边的权重都是无穷大
# # 假设节点0和节点1之间有边，权重为1
# ADJ[0, 1] = 1
# ADJ[1, 0] = 1
#
#
# # 创建支持张量
# SUP = np.full((V, V, V), np.inf)  # 初始时所有的支持成本都是无穷大
# # 假设代理1在节点0，可以支持节点1到节点2的转移，降低成本为0.5
# SUP[0, 1, 2] = 0.5
# SUP[0, 2, 1] = 0.5
#
# # 将所有的部分合并为一个状态
# S = (P.flatten(), ADJ, SUP)
#
# print("Agent positions (one-hot):")
# print(P)
# print("Adjacency matrix:")
# print(ADJ)
# print("Support tensor:")
# print(SUP)
#
# print("State:",S)

#-----------------------------------------------------------------------------
# import numpy as np
# import gym
# from gym import spaces
# import networkx as nx
# from networkx import from_numpy_array
# import matplotlib.pyplot as plt
#
#
# class GraphEnv(gym.Env):
#     def __init__(self, num_agents, adjacency_matrix, risk_levels):
#         super(GraphEnv, self).__init__()
#         self.num_agents = num_agents
#         self.adjacency_matrix = adjacency_matrix
#         self.risk_levels = risk_levels  # Risk levels for each edge
#         self.action_space = spaces.MultiDiscrete([len(adjacency_matrix)] * num_agents)
#         self.observation_space = spaces.MultiDiscrete([len(adjacency_matrix)] * num_agents)
#         self.state = np.zeros(num_agents, dtype=int)  # Start positions for agents
#
#     def step(self, actions):
#         rewards = 0
#         for i, action in enumerate(actions):
#             current_position = self.state[i]
#             if self.adjacency_matrix[current_position, action] == 1:  # Valid move
#                 self.state[i] = action
#                 edge_risk = self.risk_levels[current_position, action]
#                 rewards -= edge_risk  # Negative reward for high risk
#             else:
#                 rewards -= 10  # Large negative reward for invalid move
#
#         done = np.all(self.state == len(self.adjacency_matrix) - 1)  # Check if all agents are at the goal
#         info = {}
#         return self.state.copy(), rewards, done, info
#
#     def reset(self):
#         self.state = np.zeros(self.num_agents, dtype=int)  # Reset all agents to the start position
#         return self.state.copy()
#
#     def render(self, mode='human'):
#         G = from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
#         pos = nx.spring_layout(G)
#         nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=8,
#                 font_weight='bold')
#         nx.draw_networkx_nodes(G, pos, nodelist=self.state, node_color='red')
#         plt.show()
#
#
# # Define a simple 4-node graph with adjacency matrix and risk levels
# adj_matrix = np.array([[0, 1, 0, 0],
#                        [0, 0, 1, 0],
#                        [0, 0, 0, 1],
#                        [0, 0, 0, 0]])
# risk_levels = np.array([[0, 5, 0, 0],
#                         [0, 0, 3, 0],
#                         [0, 0, 0, 1],
#                         [0, 0, 0, 0]])
#
# env = GraphEnv(num_agents=2, adjacency_matrix=adj_matrix, risk_levels=risk_levels)
# env.reset()
# env.render()
#
# # Example step with random actions
# actions = [1, 2]  # Assuming agents can move to these positions
# state, reward, done, _ = env.step(actions)
# print(f"State: {state}, Reward: {reward}")
#

#-----------------------------------------------------------------------------------------------


import numpy as np
import gym
from gym import spaces
import networkx as nx
import matplotlib.pyplot as plt


class GraphEnv(gym.Env):
    def __init__(self, num_agents, adjacency_matrix, risk_levels, support_tensor):
        super(GraphEnv, self).__init__()
        self.num_agents = num_agents
        self.adjacency_matrix = adjacency_matrix
        self.risk_levels = risk_levels
        self.support_tensor = support_tensor
        self.num_nodes = adjacency_matrix.shape[0]

        self.action_space = spaces.MultiDiscrete([self.num_nodes + 1] * num_agents)
        self.observation_space = spaces.Dict({
            "positions": spaces.MultiBinary(self.num_nodes * num_agents),
            "adjacency": spaces.Box(low=0, high=1, shape=adjacency_matrix.shape, dtype=np.int_),
            "support": spaces.Box(low=0, high=np.inf, shape=support_tensor.shape, dtype=np.float32)
        })
        self.state = {
            "positions": np.zeros((self.num_agents, self.num_nodes), dtype=int),
            "adjacency": adjacency_matrix,
            "support": support_tensor
        }

    def step(self, actions):
        rewards = 0
        new_positions = self.state["positions"].copy()

        for i, action in enumerate(actions):
            current_position = np.argmax(self.state["positions"][i])
            if action < self.num_nodes and self.adjacency_matrix[current_position, action] == 1:
                new_positions[i] = np.zeros(self.num_nodes)
                new_positions[i, action] = 1
                rewards -= self.risk_levels[current_position, action]
            elif action == self.num_nodes:  # Support action
                rewards += 10  # example reward for successful support
            else:
                rewards -= 10  # Penalty for invalid action

        done = np.all(np.argmax(new_positions, axis=1) == self.num_nodes - 1)
        self.state["positions"] = new_positions
        return self.state.copy(), rewards, done, {}

    def reset(self):
        self.state["positions"] = np.zeros((self.num_agents, self.num_nodes), dtype=int)
        self.state["positions"][:, 0] = 1
        return self.state.copy()

    def render(self, mode='human'):
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=8,
                font_weight='bold')
        for i, agent_pos in enumerate(self.state["positions"]):
            agent_node = np.argmax(agent_pos)
            nx.draw_networkx_nodes(G, pos, nodelist=[agent_node], node_color='red', node_size=700)
        plt.show()

    def render_basic(self):
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        # 使用 nx.draw 来绘制图
        nx.draw(G, with_labels=True)
        # 显示图
        plt.show()



# Example setup and run
if __name__ == '__main__':
    num_agents = 2
    adjacency_matrix = np.array([[0, 1, 0, 1],
                                 [0, 0, 1, 0],
                                 [1, 0, 0, 1],
                                 [1, 0, 0, 0]])
    risk_levels = np.array([[0, 5, 0, 0],
                            [0, 0, 3, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]])
    support_tensor = np.zeros((4, 4, 4))  # Simplified for demo

    env = GraphEnv(num_agents, adjacency_matrix, risk_levels, support_tensor)
    state = env.reset()
    # env.render()
    env.render_basic()

    # Execute some actions
    actions = [1, 2]  # Agents attempt to move to node 1 and 2
    state, reward, done, _ = env.step(actions)
    print(f"State after actions: {state}, Reward: {reward}")
    env.render()
