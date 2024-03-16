import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MouseGCN(nn.Module):
    def __init__(self, input_dim, n_latent_features, num_gcn_layers=1, dropout=0.0, activation=nn.Tanh()):
        """
        Initialize the GraphConvNet.

        Args:
            input_dim (int): Dimensionality of input features.
            n_latent_features (int): Dimensionality of hidden layers.
            num_gcn_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
            activation (torch.nn.Module): Activation function.
        """
        super(MouseGCN, self).__init__()

        self.gcn_layers = nn.ModuleList()
        # Graph Convolutional Layers
        for i in range(num_gcn_layers):
            if i == 0:
                self.gcn_layers.append(GCNConv(input_dim, n_latent_features))
            else:
                self.gcn_layers.append(GCNConv(n_latent_features, n_latent_features))

        # Fully Connected Layers
        self.fc1 = nn.Linear(n_latent_features, n_latent_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index):
        # Apply Graph Convolutional Layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x