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
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features=prev_size, out_features=int(hidden_size)))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)