import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import time


class GridStyle_Graph_Generator:
    '''
    1) Nodes Initialization: The graph starts with n nodes.
    2) Columns and Rows Initialization: The graph is a 2D grid graph with rows and columns.
    '''
    def __init__(self, N, rows, cols):
        self.N = N
        self.rows = rows
        self.cols = cols
        self.GS_G = None # Grid Style Graph

    # 2D grid graph
    def create_grid_graph(self):
        # Create a 2D grid graph
        G = nx.grid_2d_graph(self.rows, self.cols)
        # Convert node labels from (x,y) to a continuous range of integers
        G = nx.convert_node_labels_to_integers(G)
        self.GS_G = G
        return self.GS_G
    
    def increment_node_labels(self):
        # Create a mapping from each node to its label+1
        mapping = {node: node + 1 for node in self.GS_G.nodes()}
        # Relabel nodes according to the mapping
        GS_G_relabelled = nx.relabel_nodes(self.GS_G, mapping)
        return GS_G_relabelled
    
    # Plot the graph
    def plot_graph(self):
        plt.figure(figsize=(8, 5))
        # Using nx.spring_layout for positioning nodes, with the incremented graph
        pos = nx.spring_layout(self.GS_G, seed=42)
        nx.draw(self.GS_G, pos, with_labels=True, node_color='lightgreen', edge_color='gray')
        plt.title(f"Grid Style Graph: {self.rows}x{self.cols}")
        # Save the plot
        plt.savefig(f'./grid_style/plots/grid_style_N{self.N}.png')
        # Show the plot
        plt.show()

  

