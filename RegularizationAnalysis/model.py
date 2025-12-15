import torch
import torch.nn as nn

class IdentificationNet(nn.Module):
    def __init__(self, use_dropout=False, use_bn=False):
        super().__init__()

        self.num_class = 10
        self.use_dropout = use_dropout
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=0, bias=False
        )

        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, self.num_class)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        if self.use_bn:
            x = self.bn1(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.maxpool2d(x)
        if self.use_bn:
            x = self.bn2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        if self.use_bn:
            x = self.bn3(x)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc2(x)
        return x