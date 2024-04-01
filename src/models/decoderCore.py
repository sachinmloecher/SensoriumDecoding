import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_size, downsampled_input, output_size, dropout_prob=0.1, activation=nn.ReLU(), layerNorm=False):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.downsampled_input = downsampled_input
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.layerNorm = layerNorm

        # Downsample input first
        self.input_layer = nn.Linear(input_size, downsampled_input)
        if layerNorm:
            self.input_layer_norm = nn.LayerNorm(downsampled_input)
        else:
            self.input_layer_norm = None
        self.input_dropout = nn.Dropout(dropout_prob)

        # Define upsampling ConvTranspose layers
        
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(128, 64, kernel_size=(3,6), stride=2, padding=0),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,6), stride=2, padding=0),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,6), stride=2, padding=0),
            nn.ConvTranspose2d(16, 8, kernel_size=(6,9), stride=1, padding=0),
            nn.ConvTranspose2d(8, 4, kernel_size=(6,8), stride=1, padding=0),
            
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(16),
            nn.BatchNorm2d(8),
            nn.BatchNorm2d(4),
        ])
        self.output_conv = nn.ConvTranspose2d(4, 1, kernel_size=(4,6), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Downsample the input
        x = self.input_layer(x)
        if self.input_layer_norm is not None:
            x = self.input_layer_norm(x)
        x = self.input_dropout(x)
        x = self.activation(x)
        batch_size = x.shape[0]
        # Reshape to (batch_size, channels, height, width)
        x = x.view(batch_size, 128, 2, 2)
 
        # Forward pass through ConvTranspose layers
        for conv_layer, bn_layer in zip(self.layers, self.bn_layers):
            x = conv_layer(x)
            if self.layerNorm:
                x = bn_layer(x)
            x = self.dropout(x)
            x = self.activation(x)
        x = self.output_conv(x)
        return x
    
