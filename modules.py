from torch import nn
import torch
import torch.nn.functional as F


class PreActivationBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, padding=1, dropout=None):
        super(PreActivationBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.dropout = dropout
        if dropout is not None:
            self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.bn(x)
        x = F.elu(x, inplace=True)
        x = self.conv(x)

        if self.dropout is not None:
            x = self.drop(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = PreActivationBlock(in_channels, channels)
        self.conv2 = PreActivationBlock(in_channels, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual

        return x


class ResidualModule(nn.Module):
    def __init__(self, depth, channels):
        super(ResidualModule, self).__init__()

        layers = []
        for i in range(depth):
            layers.append(ResidualBlock(channels, channels))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)

        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, channels, dropout=None):
        super(TransitionBlock, self).__init__()
        self.layer = PreActivationBlock(in_channels, channels, kernel_size=1, padding=0, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, k, dropout=None):
        super(BottleneckBlock, self).__init__()

        self.bottleneck = PreActivationBlock(in_channels, 4*k, 1, 0, dropout)
        self.layer = PreActivationBlock(4*k, k, dropout=dropout)

    def forward(self, x):
        residual = x
        x = self.bottleneck(x)
        x = self.layer(x)

        return torch.cat([residual, x], 1)


class DenseModule(nn.Module):
    def __init__(self, in_channels, depth, k, dropout=None):
        super(DenseModule, self).__init__()
        layers = []
        channels = in_channels
        for i in range(depth):
            layers.append(BottleneckBlock(channels + i*k, k, dropout=dropout))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)

        return x