import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader


def visualize_network():
    # Configuration
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10))

    # Create 3D axis
    ax_3d = fig.add_subplot(111, projection="3d")

    # Generate synthetic data for 2000 nodes
    num_nodes = 2000
    node_ids = np.arange(num_nodes)

    # Create anomaly array (5% anomalies)
    anomaly = np.zeros(num_nodes, dtype=int)
    num_anomalies = int(0.05 * num_nodes)
    anomaly[np.random.choice(node_ids, size=num_anomalies, replace=False)] = 1

    # Split into train/val/test (70/15/15)
    idx = np.arange(num_nodes)
    train_idx, testval_idx = train_test_split(idx, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, random_state=42)

    # Create random positions
    x = np.random.rand(num_nodes) * 100
    y = np.random.rand(num_nodes) * 100
    z = np.random.rand(num_nodes) * 10

    # Create graph structure
    G = nx.Graph()
    for i in idx:
        split = "train" if i in train_idx else ("val" if i in val_idx else "test")
        G.add_node(i, pos=(x[i], y[i], z[i]), anomaly=anomaly[i], split=split)

    # Create edges (connect to 3-5 nearest neighbors)
    for i in range(num_nodes):
        distances = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2 + (z - z[i]) ** 2)
        neighbors = np.argsort(distances)[1:6]  # 5 closest neighbors
        for j in neighbors:
            G.add_edge(i, j)

    # Visualization parameters
    split_colors = {"train": "blue", "val": "orange", "test": "green"}
    anomaly_colors = {0: "lime", 1: "red"}

    # Draw edges
    for split in ["train", "val", "test"]:
        split_nodes = [n for n in G.nodes if G.nodes[n]["split"] == split]
        subgraph = G.subgraph(split_nodes)

        for edge in subgraph.edges():
            x0, y0, z0 = G.nodes[edge[0]]["pos"]
            x1, y1, z1 = G.nodes[edge[1]]["pos"]
            ax_3d.plot(
                [x0, x1],
                [y0, y1],
                [z0, z1],
                color=split_colors[split],
                alpha=0.2,
                linewidth=0.5,
            )

    # Draw nodes
    for node in G.nodes():
        x, y, z = G.nodes[node]["pos"]
        split = G.nodes[node]["split"]
        anomaly_status = G.nodes[node]["anomaly"]

        ax_3d.scatter(
            x,
            y,
            z,
            c=anomaly_colors[anomaly_status],
            s=20,
            edgecolors=split_colors[split],
            linewidths=0.5,
            alpha=0.8,
        )

    # Configure 3D view
    ax_3d.view_init(elev=30, azim=45)
    ax_3d.set_axis_off()
    ax_3d.set_title("3D Network Visualization with Anomaly Detection", pad=20)

    plt.tight_layout()
    plt.savefig("network_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

    return G


if __name__ == "__main__":
    graph = visualize_network()
