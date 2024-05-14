import torch
import torch.nn as nn

class MLP_CIFAR10(nn.Module):
    """A simple multilayer perceptron model."""
    def __init__(self):
        super(MLP_CIFAR10, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(3*32*32, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

class CNN_CIFAR10(nn.Module):
    """A convolutional neural network model with 3 convolutional layers."""
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.Maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.Maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.Maxpool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Maxpool1(x)
        x = self.conv2(x)
        x = self.Maxpool2(x)
        x = self.conv3(x)
        x = self.Maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class CNN_deep_CIFAR10(nn.Module):
    """A deep convolutional neural network model with increased depth and complexity."""
    def __init__(self):
        super(CNN_deep_CIFAR10, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.Maxpool1 = nn.MaxPool2d(2)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Maxpool2 = nn.MaxPool2d(2)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.Maxpool3 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.Maxpool1(x)
        x = self.conv_block2(x)
        x = self.Maxpool2(x)
        x = self.conv_block3(x)
        x = self.Maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class MLP_MNIST(nn.Module):
    """A simple multilayer perceptron model."""
    def __init__(self):
        super(MLP_CIFAR10, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(3*32*32, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1,12,3, bias=False, padding=1),
            nn.BatchNorm2d(12),
            nn.Sigmoid(),
            nn.Conv2d(12,24,3, bias=False, padding=1),
            nn.BatchNorm2d(24),
            nn.Sigmoid(),
            nn.Conv2d(24,36,3, bias=False, padding=1),
            nn.BatchNorm2d(36),
            nn.Sigmoid()
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(36*7*7,10, bias=False)


    def forward(self, x):
        x = self.conv_block(x)
        x = self.maxpool1(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

class CNN_STL10(nn.Module):
    """A convolutional neural network model with 3 convolutional layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.Maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.Maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.Maxpool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 12 * 12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Maxpool1(x)
        x = self.conv2(x)
        x = self.Maxpool2(x)
        x = self.conv3(x)
        x = self.Maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x