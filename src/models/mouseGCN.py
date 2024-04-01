import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MouseGCN(nn.Module):
    def __init__(self, input_dim, hidden_size, n_latent_features, num_gcn_layers=1, dropout=0.0, activation=nn.Tanh()):
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
                self.gcn_layers.append(GCNConv(input_dim, hidden_size))
            else:
                self.gcn_layers.append(GCNConv(hidden_size, hidden_size))

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, n_latent_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, batch):
        # Apply Graph Convolutional Layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = self.dropout(x)
            x = self.activation(x)
            
        
        x = global_mean_pool(x, batch)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)

        return x