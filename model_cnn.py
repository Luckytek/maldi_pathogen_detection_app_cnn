import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)    # Convolutional layer
        out = self.bn1(out)    # Batch normalization
        out = self.relu(out)   # Activation function

        out = self.conv2(out)
        out = self.bn2(out)

        # If downsample is not None, adjust the identity mapping
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the identity (skip connection)
        out += identity
        out = self.relu(out)

        return out

class MALDIResNet(nn.Module):
    def __init__(self, num_classes):
        super(MALDIResNet, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer
        self.conv = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(ResidualBlock, out_channels=64, blocks=2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, out_channels=512, blocks=2, stride=2)

        # Adaptive pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a residual layer composed of several residual blocks.

        Parameters:
        - block: class
            The residual block class.
        - out_channels: int
            The number of output channels for the blocks in this layer.
        - blocks: int
            The number of residual blocks in this layer.
        - stride: int
            The stride for the first block in this layer.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # Adjust the identity mapping if output dimensions change
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        # First block may change the dimensions (e.g., stride, channels)
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels  # Update in_channels for next blocks

        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: (batch_size, 1, sequence_length)
        x = self.conv(x)      # Initial convolution
        x = self.bn(x)        # Batch normalization
        x = self.relu(x)      # Activation function
        x = self.maxpool(x)   # Max pooling

        x = self.layer1(x)    # Residual layer 1
        x = self.layer2(x)    # Residual layer 2
        x = self.layer3(x)    # Residual layer 3
        x = self.layer4(x)    # Residual layer 4

        x = self.avgpool(x)   # Adaptive average pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)        # Fully connected layer

        return x
