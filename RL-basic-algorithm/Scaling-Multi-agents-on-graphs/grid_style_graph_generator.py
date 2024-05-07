import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import time
from gs_rg import GridStyle_Graph_Generator


class TCGRE_GridStyle_Graph_Generator:
    def __init__(self, N, rows, cols, risk_edge_ratio=0.2):
        self.N = N
        self.rows = rows
        self.cols = cols
        self.risk_edge_ratio = risk_edge_ratio
        self.TCGRE_G = None
        self.risk_edges_with_support_nodes = None

    def create_gridstyle_graph(self):
        grid_graph = GridStyle_Graph_Generator(self.N, self.rows, self.cols)
        grid_graph.create_grid_graph()
        grid_graph_incremented = grid_graph.increment_node_labels()
        self.TCGRE_G = grid_graph_incremented
        print("Grid Style Graph created...")
        return self.TCGRE_G

    # just pick the shortest path
    def pick_edges_on_shortest_path(self):
        source = list(self.TCGRE_G.nodes())[0]
        target = len(self.TCGRE_G.nodes())

        print(f"Source: {source}, Target: {target}")
        # Find all shortest paths between source and target
        all_shoretes_paths = list(nx.all_shortest_paths(self.TCGRE_G, source=source, target=target, weight='weight'))

        # Extract edges from the path (consecutive pairs of nodes)
        edges_list = []
        for path in all_shoretes_paths:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            edges_list.append(edges)

        # Extra unique edges from all paths
        unique_edges = set()
        for path in all_shoretes_paths:
            # Extract edges from the path (consecutive pairs of nodes) and add them to the set
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            unique_edges.update(edges)

        return len(all_shoretes_paths), list(unique_edges)

    # dont add cost to the edges, just pick the risk edges and support nodes
    def pick_risk_edges_and_support_nodes(self):
        print("Picking risk edges and support nodes...")
        # Pick random edges as risk edges from edges, it should be 0.2 of the total edges
        # Calculate the number of edges to select as risky
        num_risk_edges = int(len(self.TCGRE_G.edges()) * self.risk_edge_ratio)

        # Randomly select edges without replacement
        ## one way: add at least some risk edges on the shortest path with other edges
        len_all_paths, unique_edges_on_shortest_path = self.pick_edges_on_shortest_path()
        risk_edges = random.sample(self.TCGRE_G.edges(), num_risk_edges - 1)
        # Filter out edges that are already in risk_edges
        available_edges = [edge for edge in unique_edges_on_shortest_path if edge not in risk_edges]
        ## add the edge on the shortest path
        # Check if there are any available edges to add
        if available_edges:
            # Randomly select an edge that's not already a risk edge
            chosen_edge = random.choice(available_edges)
            print(f"chosen_edge: {chosen_edge}")
            # Add this edge to risk_edges
            risk_edges.append(chosen_edge)

        ## pick up neighbors of the risk edges as support nodes
        ## check if node is in the neighbors of the risk edges

        risk_edge_with_support_nodes = {}
        support_nodes_used = set()
        for edge in risk_edges:
            total_neighbors = list(self.TCGRE_G.neighbors(edge[0])) + list(self.TCGRE_G.neighbors(edge[1]))
            ## only pick the neighbors that are not used as support nodes before
            for neighbor in total_neighbors:
                if neighbor in support_nodes_used:
                    total_neighbors.remove(neighbor)
            random_support_node = random.choice(total_neighbors)
            risk_edge_with_support_nodes[edge] = (random_support_node,)
            # update the support nodes used
            support_nodes_used.add(random_support_node)

            print(f"risk_edge_with_support_nodes: {risk_edge_with_support_nodes}")
        time.sleep(2)
        self.risk_edges = risk_edge_with_support_nodes
        return self.risk_edges

    #  add cost to the edges including the risk edges
    def add_cost_to_edges(self):
        print("Adding cost to the edges...")
        for edge in self.TCGRE_G.edges():
            if edge in self.risk_edges.keys():
                print(f"risk_edge: {edge}, support_nodes: {self.risk_edges[edge][0]}")
                self.TCGRE_G[edge[0]][edge[1]]['cost'] = [20, (self.risk_edges[edge][0],)]
            else:
                print(f"normal_edge: {edge}")
                # either fixed cost for normal edges, lesser than the risk edge cost
                # self.TCGRE_G[edge[0]][edge[1]]['cost'] = 5
                # or random cost for normal edges, between 1 and 10, lesser than the risk edge cost
                self.TCGRE_G[edge[0]][edge[1]]['cost'] = random.randint(1, 10)

        return self.TCGRE_G

    # convert the graph to compatible graph
    def convert_to_compatible_graph(self):
        print("Converting to compatible graph...")
        nodes = {node: {} for node in self.TCGRE_G.nodes()}
        for edge in self.TCGRE_G.edges():
            # Unpack the edge nodes
            node1, node2 = edge
            nodes[node1][node2] = self.TCGRE_G[node1][node2]['cost']  # For node1 -> node2
            nodes[node2][node1] = self.TCGRE_G[node1][node2]['cost']  # For node2 -> node1
        return nodes

    # plot the graph
    def plot_graph(self):
        plt.figure()
        # Using nx.spring_layout for positioning nodes, with the incremented graph
        pos = nx.spring_layout(self.TCGRE_G, seed=42)
        nx.draw(self.TCGRE_G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray',
                font_color='w')
        nx.draw_networkx_edge_labels(self.TCGRE_G, pos,
                                     edge_labels={(u, v): d['cost'] for u, v, d in self.TCGRE_G.edges(data=True)})
        # change color to red for the risk edges
        nx.draw_networkx_edges(self.TCGRE_G, pos, edgelist=self.risk_edges.keys(), edge_color='red', width=1.0)
        plt.title(f"TCGRE Grid Style Graph: {self.rows}x{self.cols}")
        # Save the plot
        plt.savefig(f'./grid_style/plots/tcgre_grid_N{self.N}.png')
        # Show the plot
        plt.show()


'''
# Example grid graphs
# Grid: 2x5, 10 nodes, node labels: 1 - 10
# Grid: 3x5, 15 nodes, node labels: 1 - 15
# Grid: 4x5, 20 nodes, node labels: 1 - 20
# Grid: 5x5, 25 nodes, node labels: 1 - 25
# Grid: 6x5, 30 nodes, node labels: 1 - 30
'''
# Parameters
N = 10  # Number of nodes
cols = 5  # Number of columns
rows, cols = N // cols, cols  # Number of rows and columns
risk_edge_ratio = 0.2  # 20% of the edges are risk edges

# Create a TCGRE Grid Style Graph
tcgre_gs = TCGRE_GridStyle_Graph_Generator(N, rows, cols, risk_edge_ratio)
tcgre_gs.create_gridstyle_graph()
tcgre_gs.pick_risk_edges_and_support_nodes()
tcgre_gs.add_cost_to_edges()
tcgre_gs.plot_graph()  # plot the graph
graph_info_tcgre_gs = tcgre_gs.convert_to_compatible_graph()
print(f"Graph Info: {graph_info_tcgre_gs}")