import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def visualize_network():
    # Load real graph data
    graph_path = "data/processed/combined_graph.pt"
    data = torch.load(graph_path, weights_only=False)

    # Use first 3 features as positions (if available)
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    num_nodes = x.shape[0]
    if x.shape[1] < 3:
        # Not enough features for 3D, generate random positions
        pos = np.random.rand(num_nodes, 3) * 100
    else:
        pos = x[:, :3]

    # Assign splits
    split = np.array(["train"] * num_nodes)
    split[data.val_mask.cpu().numpy()] = "val"
    split[data.test_mask.cpu().numpy()] = "test"

    # Build networkx graph from edge_index
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, pos=tuple(pos[i]), anomaly=int(y[i]), split=split[i])
    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        G.add_edge(int(src), int(dst))

    # Visualization
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10))
    ax_3d = fig.add_subplot(111, projection="3d")

    split_colors = {"train": "blue", "val": "orange", "test": "green"}
    anomaly_colors = {0: "lime", 1: "red"}

    # Draw edges
    for split_name in ["train", "val", "test"]:
        split_nodes = [n for n in G.nodes if G.nodes[n]["split"] == split_name]
        subgraph = G.subgraph(split_nodes)
        for edge in subgraph.edges():
            x0, y0, z0 = G.nodes[edge[0]]["pos"]
            x1, y1, z1 = G.nodes[edge[1]]["pos"]
            ax_3d.plot([x0, x1], [y0, y1], [z0, z1], color=split_colors[split_name], alpha=0.2, linewidth=0.5)

    # Draw nodes
    for node in G.nodes():
        x_, y_, z_ = G.nodes[node]["pos"]
        split_ = G.nodes[node]["split"]
        anomaly_status = G.nodes[node]["anomaly"]
        ax_3d.scatter(x_, y_, z_, c=anomaly_colors[anomaly_status], s=20, edgecolors=split_colors[split_], linewidths=0.5, alpha=0.8)

    ax_3d.view_init(elev=30, azim=45)
    ax_3d.set_axis_off()
    ax_3d.set_title("3D Network Visualization with Anomaly Detection (Real Data)", pad=20)
    plt.tight_layout()
    plt.savefig("network_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()
    return G


if __name__ == "__main__":
    graph = visualize_network()
