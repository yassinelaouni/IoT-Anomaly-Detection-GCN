import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCNAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super(GCNAnomalyDetector, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Première couche GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Deuxième couche GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
