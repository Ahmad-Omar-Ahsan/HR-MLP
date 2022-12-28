import torch
import torch.nn.functional as F
import torch.nn as nn


class block(nn.Module):
    """Blocks in the ResNet architecture

    Args:
        nn (nn.Module): Base class for all neural network modules.
    """

    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        """Initializes block for ResNet

        Args:
            in_channels (int): Input channels
            intermediate_channels (int): Number of channels for intermediate layers
            identity_downsample (nn.Moduel, optional): Conv layer to downsample size of the input. Defaults to None.
            stride (int, optional): Value of stride. Defaults to 1.
        """
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        """Forward function for the block module

        Args:
            x (torch.tensor): Input tensor

        Returns:
            [torch.tensor]: Output tensor
        """
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """ResNet architecture

    Args:
        nn (nn.Module): Base class for all neural network modules.
    """

    def __init__(self, layers, image_channels, num_classes):
        """Initializes ResNet

        Args:

            layers (list): List of number of blocks in each layer
            image_channels (int): Number of channels in the image
            num_classes (int): Number of total classes
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        """Forward function for ResNet

        Args:
            x (torch.tensor): Input tensor

        Returns:
            [torch.tensor]: Output tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, num_residual_blocks, intermediate_channels, stride):
        """This function makes the layer for the ResNet model in an iterative manner

        Args:
            num_residual_blocks (int): number of residual blocks
            intermediate_channels (int): value of intermediate channels
            stride (int): stride size

        Returns:
            nn.Module : Layer architecture
        """
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
