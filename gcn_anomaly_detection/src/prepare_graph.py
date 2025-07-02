import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import os
from prepare_data import load_and_split_data, preprocess_data


def build_graph_for_split(features, labels, k_neighbors=3):
    """Build graph for a single split"""
    x = torch.tensor(features, dtype=torch.float)
    y = (
        torch.tensor(labels.values, dtype=torch.long)
        if hasattr(labels, "values")
        else torch.tensor(labels, dtype=torch.long)
    )

    # KNN graph construction
    if len(features) > k_neighbors:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(features)
        _, indices = nbrs.kneighbors(features)

        edge_list = []
        for i in range(len(features)):
            for j in indices[i, 1:]:  # Skip self-connection
                edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Fallback for small graphs
        edge_index = torch.combinations(torch.arange(len(features)), 2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Undirected

    return Data(x=x, edge_index=edge_index, y=y)


def save_graphs(train_graph, val_graph, test_graph, output_dir="data/processed"):
    """Save all split graphs"""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_graph, os.path.join(output_dir, "train_graph.pt"))
    torch.save(val_graph, os.path.join(output_dir, "val_graph.pt"))
    torch.save(test_graph, os.path.join(output_dir, "test_graph.pt"))
    print(f"Graphs saved to {output_dir}")


def combine_graphs(train_graph, val_graph, test_graph):
    """Combine all graphs into one with proper indices"""
    n_train = train_graph.x.shape[0]
    n_val = val_graph.x.shape[0]

    # Adjust indices
    val_edge_index = val_graph.edge_index + n_train
    test_edge_index = test_graph.edge_index + n_train + n_val

    # Combine everything
    combined_graph = Data(
        x=torch.cat([train_graph.x, val_graph.x, test_graph.x], dim=0),
        edge_index=torch.cat(
            [train_graph.edge_index, val_edge_index, test_edge_index], dim=1
        ),
        y=torch.cat([train_graph.y, val_graph.y, test_graph.y], dim=0),
        train_mask=torch.arange(n_train),
        val_mask=torch.arange(n_train, n_train + n_val),
        test_mask=torch.arange(
            n_train + n_val, n_train + n_val + test_graph.x.shape[0]
        ),
    )
    return combined_graph


if __name__ == "__main__":
    data_path = "data/synthetic_iot_dataset.csv"
    train_df, val_df, test_df = load_and_split_data(data_path)

    if train_df is not None:
        # Preprocess and build graphs
        X_train, y_train, train_ids, scaler = preprocess_data(train_df)
        train_graph = build_graph_for_split(X_train, y_train)

        X_val, y_val, val_ids, _ = preprocess_data(val_df, scaler)
        val_graph = build_graph_for_split(X_val, y_val)

        X_test, y_test, test_ids, _ = preprocess_data(test_df, scaler)
        test_graph = build_graph_for_split(X_test, y_test)

        # Save individual graphs
        save_graphs(train_graph, val_graph, test_graph)

        # Optional: Combine into one graph with masks
        combined_graph = combine_graphs(train_graph, val_graph, test_graph)
        torch.save(combined_graph, "data/processed/combined_graph.pt")
        print("Combined graph saved with train/val/test masks")
