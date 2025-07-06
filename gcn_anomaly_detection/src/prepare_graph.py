import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import os
from prepare_data import load_and_split_data, preprocess_data


def build_graph_for_split(features, labels, k_neighbors=8):
    """Construire un graphe pour un seul ensemble de données (train/val/test)"""
    x = torch.tensor(features, dtype=torch.float)
    y = (
        torch.tensor(labels.values, dtype=torch.long)
        if hasattr(labels, "values")
        else torch.tensor(labels, dtype=torch.long)
    )

    # Construction du graphe KNN
    if len(features) > k_neighbors:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(features)
        _, indices = nbrs.kneighbors(features)

        edge_list = []
        for i in range(len(features)):
            for j in indices[i, 1:]:  # Ignorer l'auto-connexion
                edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Solution de repli pour les petits graphes
        edge_index = torch.combinations(torch.arange(len(features)), 2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Non dirigé

    return Data(x=x, edge_index=edge_index, y=y)


def save_graphs(train_graph, val_graph, test_graph, output_dir="data/processed"):
    """Sauvegarder tous les graphes séparés"""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_graph, os.path.join(output_dir, "train_graph.pt"))
    torch.save(val_graph, os.path.join(output_dir, "val_graph.pt"))
    torch.save(test_graph, os.path.join(output_dir, "test_graph.pt"))
    print(f"Graphes sauvegardés dans {output_dir}")


def combine_graphs(train_graph, val_graph, test_graph):
    """Combiner tous les graphes en un seul avec des indices appropriés"""
    n_train = train_graph.x.shape[0]
    n_val = val_graph.x.shape[0]

    # Ajuster les indices
    val_edge_index = val_graph.edge_index + n_train
    test_edge_index = test_graph.edge_index + n_train + n_val

    # Combiner tout
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
    # Chemin vers le fichier de données
    data_path = "data/synthetic_iot_dataset.csv"
    train_df, val_df, test_df = load_and_split_data(data_path)

    if train_df is not None:
        # Prétraiter et construire les graphes
        X_train, y_train, scaler = preprocess_data(train_df)
        train_graph = build_graph_for_split(X_train, y_train)

        X_val, y_val, _ = preprocess_data(val_df, scaler)
        val_graph = build_graph_for_split(X_val, y_val)

        X_test, y_test, _ = preprocess_data(test_df, scaler)
        test_graph = build_graph_for_split(X_test, y_test)

        # Sauvegarder les graphes individuels
        save_graphs(train_graph, val_graph, test_graph)

        # Optionnel: Combiner en un seul graphe avec des masques
        combined_graph = combine_graphs(train_graph, val_graph, test_graph)
        torch.save(combined_graph, "data/processed/combined_graph.pt")
        print("Graphe combiné sauvegardé avec les masques train/val/test")
