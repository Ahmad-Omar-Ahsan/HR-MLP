import torch
from torch import nn


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleConvNet, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )

        # Define adaptive pooling layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))  # Output size (32, 32)

        # Define fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 32, 128)
        self.fc2 = nn.Linear(
            128, num_classes
        )  # 10 is the number of output classes (adjust as needed)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and adaptive pooling
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor before fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    image_tensor = torch.randn((3, 3, 32, 32))
    mixer = SimpleConvNet(num_classes=100)
    output = mixer(image_tensor)
    print(output.shape)
    print(count_parameters(mixer))
