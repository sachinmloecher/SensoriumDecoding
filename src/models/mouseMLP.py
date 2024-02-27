import torch.nn as nn

class MouseMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.1, activation=nn.ReLU()):
        super(MouseMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_sizes)
        self.activation = activation

        layers = []
        layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[0]))
        layers.append(self.activation)
        
        # Hidden Layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(in_features=hidden_sizes[i-1], out_features=hidden_sizes[i]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))

        # Output Layer
        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)