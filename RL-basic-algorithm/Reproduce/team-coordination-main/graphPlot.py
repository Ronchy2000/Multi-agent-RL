import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt   
import networkx as nx
from matplotlib.pyplot import figure 
import numpy as np
import re

def plot_joint_state_space_graph(S,D,E):
    G = nx.Graph(node_type="joint_state_space")
    G.add_edges_from(E)
    print("Nodes")
    print([node for node in G])
    # specify the nodes you want here
    print("Source, Destination")
    print(S,D)
    color_map = []
    for node in G:
        color_map.append('green')
        # if node==str(S):
        #     color_map.append('blue')
        # elif node==str(D):
        #     color_map.append('red')
        # else:
        #     color_map.append('green')
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    #pos = nx.spectral_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                            node_color = color_map, node_size = 70)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=E, edge_color='black', arrows=False, edge_size=5)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    #figure(figsize=(50, 50), dpi=80)

def plot_environment_graph(E):
    G = nx.Graph(node_type="environment_graph")
    G.add_weighted_edges_from(E)
    weight = nx.get_edge_attributes(G, 'weight')
    print(G.edges)
    print(G.nodes)
    print(weight)
    pos = nx.spring_layout(G)
    # pos = nx.shell_layout(G)
    #pos = nx.spectral_layout(G)
    nx.draw_networkx(G, with_labels=True, pos=pos, node_size= 1500, node_color='red', edge_color='green', arrowsize=100, font_size=25)
    plt.show()   
    
def plot_joint_state_space_graphTT(S,D,E):
    G = nx.Graph(node_type="joint_state_space")
    G.add_edges_from(E)
    nodes = [eval(node) for node in G]
    print([eval(node) for node in G])

    pos = {}
    for n in nodes:
        pos[str(n)] = n
        
    print(pos)

    plt.figure()
    #nx.draw(graph, with_labels=False)
    nx.draw(G, with_labels=False, pos=pos,node_color='green', node_size = 70)
    nx.draw_networkx_labels(G, pos)
    plt.show()
    
def plot_environment_graph_test(E):
    G = nx.grid_2d_graph(4, 4)  # 4x4 grid

    pos = nx.spring_layout(G, iterations=1000, seed=39775)

    # Create a 2x2 subplot
    fig, all_axes = plt.subplots(2, 2)
    ax = all_axes.flat

    nx.draw(G, pos, ax=ax[0], font_size=8)
    nx.draw(G, pos, ax=ax[1], node_size=0, with_labels=False)
    nx.draw(
        G,
        pos,
        ax=ax[2],
        node_color="tab:green",
        edgecolors="tab:gray",  # Node surface color
        edge_color="tab:gray",  # Color of graph edges
        node_size=250,
        with_labels=False,
        width=6,
    )
    H = G.to_directed()
    nx.draw(
        H,
        pos,
        ax=ax[3],
        node_color="tab:orange",
        node_size=20,
        with_labels=False,
        arrowsize=10,
        width=2,
    )

    # Set margins for the axes so that nodes aren't clipped
    for a in ax:
        a.margins(0.10)
    fig.tight_layout()
    plt.show()
