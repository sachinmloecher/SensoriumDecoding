import torch.nn as nn

class BehaviourMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.1, activation=nn.ReLU()):
        super(BehaviourMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.activation = activation

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)