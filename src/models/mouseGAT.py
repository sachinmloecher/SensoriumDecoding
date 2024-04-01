import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

class MouseGAT(nn.Module):
    def __init__(self, input_dim, hidden_size, n_latent_features, num_gat_layers=1, dropout=0.0, activation=nn.Tanh()):
        """
        Initialize the GraphConvNet.

        Args:
            input_dim (int): Dimensionality of input features.
            n_latent_features (int): Dimensionality of hidden layers.
            num_gat_layers (int): Number of GAT layers.
            dropout (float): Dropout probability.
            activation (torch.nn.Module): Activation function.
        """
        super(MouseGAT, self).__init__()

        self.gat_layers = nn.ModuleList()
        # Graph Attention Layers
        for i in range(num_gat_layers):
            if i == 0:
                self.gat_layers.append(GATConv(input_dim, hidden_size, heads=2))
            else:
                self.gat_layers.append(GATConv(hidden_size * 2, hidden_size, heads=2))

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size * 2, n_latent_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, batch):

        # Apply Graph Attention Layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = torch.cat(x.split(1, dim=1), dim=-1)  # Concatenate heads
            x = self.dropout(x)
            x = self.activation(x)
        
        x = global_mean_pool(x, batch)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)

        return x