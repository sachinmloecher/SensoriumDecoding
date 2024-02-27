import torch.nn as nn

class MLP_core(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.1, activation=nn.ReLU(), layerNorm=False):
        super(MLP_core, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.activation = activation

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if layerNorm:
            layers.append(nn.LayerNorm(hidden_sizes[0]))
        layers.append(self.activation)

        # Define the hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if layerNorm:
                layers.append(nn.LayerNorm(hidden_sizes[i+1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))
        
        # Define the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)