import torch
import torch.nn as nn

class InverseVGG(nn.Module):
    def __init__(self, latent_features, output_shape=(1, 36, 64), dropout_prob=0.0, layer_norm=False):
        super(InverseVGG, self).__init__()
        self.latent_features = latent_features
        self.output_shape = output_shape

        # Define the transposed convolutional layers for upsampling
        self.trans_conv1 = InverseConvBlock(latent_features, 512, dropout_prob, layer_norm, kernel_size=4, stride=2, padding=1)
        self.trans_conv2 = InverseConvBlock(512, 256, dropout_prob, layer_norm, kernel_size=4, stride=2, padding=1)
        self.trans_conv3 = InverseConvBlock(256, 128, dropout_prob, layer_norm, kernel_size=4, stride=2, padding=1)
        self.trans_conv4 = InverseConvBlock(128, 64, dropout_prob, layer_norm, kernel_size=4, stride=2, padding=1)
        self.trans_conv5 = InverseConvBlock(64, output_shape[0], dropout_prob, layer_norm, kernel_size=3, stride=1, padding=1)

        # Define activation function for the output layer (e.g., Tanh for normalized images)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        # Reshape the latent features to match the expected input shape of the transposed convolutional layers
        x = x.view(x.size(0), self.latent_features, 1, 1)
        print(x.shape)
        # Forward pass through transposed convolutional layers with ReLU activation
        x = nn.ReLU()(self.trans_conv1(x))
        print(x.shape)
        x = nn.ReLU()(self.trans_conv2(x))
        print(x.shape)
        x = nn.ReLU()(self.trans_conv3(x))
        print(x.shape)
        x = nn.ReLU()(self.trans_conv4(x))
        print(x.shape)

        # Forward pass through the final transposed convolutional layer
        x = self.trans_conv5(x)
        print(x.shape)

        # Apply activation function for the output layer
        x = self.output_activation(x)

        return x


# Define a basic convolutional block for the inverse VGG16 architecture
class InverseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0, layer_norm=False, **kwargs):
        super(InverseConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.activation = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0.0 else None
        self.layer_norm = nn.LayerNorm(out_channels) if layer_norm else None

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x