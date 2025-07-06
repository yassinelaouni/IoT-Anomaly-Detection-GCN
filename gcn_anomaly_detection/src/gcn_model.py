import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCNAnomalyDetector(nn.Module):
    """Modèle GCN pour la détection d'anomalies dans les graphes"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super(GCNAnomalyDetector, self).__init__()
        # Première couche de convolution graphique
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Deuxième couche de convolution graphique
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Couche de classification finale
        self.classifier = nn.Linear(hidden_dim, output_dim)
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """Propagation avant du modèle"""
        x, edge_index = data.x, data.edge_index

        # Première couche GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Deuxième couche GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Classification finale
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
