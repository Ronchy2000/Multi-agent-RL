import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx


class SupportNode:
    def __init__(self, node_id: int):
        self.id = node_id
        self.supporting_agents = set()  # 用于存储在当前节点提供支持的代理集合


class SupportEdge:
    def __init__(self, source: int, target: int, nominal_cost: float, reduced_cost: float):
        self.source = source
        self.target = target
        self.nominal_cost = nominal_cost
        self.reduced_cost = reduced_cost


class SupportGraph:
    def __init__(self, num_nodes: int, start_nodes: list, goal_nodes: list):
        self.num_nodes = num_nodes
        self.start_nodes = start_nodes
        self.goal_nodes = goal_nodes
        self.nodes = {i: SupportNode(i) for i in range(num_nodes)}
        self.edges = {}  # 用于存储环境图中的边及其成本信息
        self.adjacency_matrix = np.full((num_nodes, num_nodes), np.inf)  # 初始化邻接矩阵
        self.support_tensor = np.full((num_nodes, num_nodes, num_nodes), np.inf)  # 初始化支持张量

    def add_edge(self, source: int, target: int, nominal_cost: float, support_nodes: list):
        self.edges[(source, target)] = SupportEdge(source, target, nominal_cost, nominal_cost)
        self.adjacency_matrix[source, target] = nominal_cost

        for support_node in support_nodes:
            self.support_tensor[support_node, source, target] = nominal_cost  # 初始值为未支持时的成本
            self.nodes[support_node].supporting_agents.add(target)

    def update_support_cost(self, supporting_agent: int, target_edge: tuple, reduced_cost: float):
        source, target = target_edge
        self.support_tensor[supporting_agent, source, target] = reduced_cost
        self.edges[target_edge].reduced_cost = reduced_cost

    def get_edge_cost(self, source: int, target: int, supporting_agents: set) -> float:
        if (source, target) in self.edges:
            if supporting_agents and source in supporting_agents:
                return self.support_tensor[supporting_agent, source, target]
            else:
                return self.edges[(source, target)].nominal_cost
        return np.inf  # 如果无边连接，则返回无穷大

    def get_joint_action_cost(self, agent_positions: list, actions: list) -> float:
        total_cost = 0.0
        for agent_id, (next_node, is_supporting) in enumerate(actions):
            if is_supporting:
                for edge in self.edges.values():
                    if edge.source == agent_positions[agent_id]:
                        total_cost += edge.reduced_cost
            else:
                current_node = agent_positions[agent_id]
                next_node_cost = self.get_edge_cost(current_node, next_node, set())
                total_cost += next_node_cost
        return total_cost

    def reset(self):
        for node in self.nodes.values():
            node.supporting_agents.clear()
        for edge in self.edges.values():
            edge.reduced_cost = edge.nominal_cost

    def get_joint_state(self, agent_positions: list) -> dict:
        joint_state = {
            'agent_positions': agent_positions,
            'graph_connectivity': self.adjacency_matrix.tolist(),
            'supporting_mechanism': self.support_tensor.tolist(),
        }
        return joint_state

    def perform_joint_action(self, agent_positions: list, actions: list) -> list:
        next_agent_positions = []
        for agent_id, (next_node, is_supporting) in enumerate(actions):
            if is_supporting:
                edge_key = (agent_positions[agent_id], next_node)
                if edge_key in self.edges:  # 检查边缘是否存在
                    self.update_support_cost(agent_id, edge_key, self.edges[edge_key].reduced_cost)
                else:
                    raise ValueError(f"Edge from node {agent_positions[agent_id]} to node {next_node} does not exist.")
            next_agent_positions.append(next_node)
        return next_agent_positions

    def visualize_graph(self, agent_positions: list, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        node_colors = ['lightgray' if node_id not in agent_positions else 'blue' for node_id in range(self.num_nodes)]
        pos = {node_id: (node_id, 0) for node_id in range(self.num_nodes)}  # 简化为一维排列，实际应用可自定义节点位置

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(self.num_nodes))
        nx_graph.add_edges_from([(edge.source, edge.target) for edge in self.edges.values()])

        nx.draw(nx_graph, pos=pos, node_color=node_colors, with_labels=True, ax=ax)

        return ax


def main():
    sg = SupportGraph(10, [0], [9])
    sg.add_edge(0, 1, 1.0, [1])  # 添加一条边，节点0到节点1，成本为1.0，支持节点为1
    sg.add_edge(1, 2, 2.0, [2, 3])  # 添加另一条边，节点1到节点2，成本为2.0，支持节点为2和3

    agent_positions = [0, 1, 2, 3]

    # 获取联合状态
    joint_state = sg.get_joint_state(agent_positions)

    # 执行联合动作
    actions = [(1, False), (2, False), (3, False), (5, False)]
    next_agent_positions = sg.perform_joint_action(agent_positions, actions)

    # 可视化环境图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sg.visualize_graph(agent_positions, axs[0])
    axs[0].set_title("Before Joint Action")

    sg.visualize_graph(next_agent_positions, axs[1])
    axs[1].set_title("After Joint Action")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
