"""
This file contains basic building of LeNet5 and LeNet5 half
For MNIST and FashionMNIST

Created by Kunhong Yu
Date: 2021/07/16
"""
import torch as t

class LeNet5(t.nn.Module):
    """Define LeNet5 model"""

    def __init__(self):
        super().__init__()

        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 2),
            t.nn.BatchNorm2d(6),
            t.nn.ReLU(inplace = True)
        )

        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            t.nn.BatchNorm2d(16),
            t.nn.ReLU(inplace = True)
        )

        self.fc = t.nn.Sequential(
            t.nn.Linear(400, 120),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(120, 84),
            t.nn.ReLU(inplace = True)
        )

        self.classifier = t.nn.Sequential(
            t.nn.Linear(84, 10)
        )

        self.max_pool = t.nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        """x has shape [m, 1, 28, 28]"""
        x = self.layer1(x) # [6, 28, 28]
        x = self.max_pool(x) # [6, 14, 14]
        x = self.layer2(x) # [16, 10, 10]
        x = self.max_pool(x) # [16, 5, 5]

        x = x.view(x.size(0), -1)
        x = self.fc(x) # [84]
        x = self.classifier(x)

        return x


class LeNet5Half(t.nn.Module):
    """Define LeNet5 half model
    Note: in the paper, authors did not give details of the net, we just build them from our perspective
    """

    def __init__(self):
        super().__init__()

        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 5, stride = 1, padding = 2),
            t.nn.BatchNorm2d(3),
            t.nn.ReLU(inplace = True)
        )

        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 5, stride = 1),
            t.nn.BatchNorm2d(8),
            t.nn.ReLU(inplace = True)
        )

        self.fc = t.nn.Sequential(
            t.nn.Linear(200, 60),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(60, 42),
            t.nn.ReLU(inplace = True)
        )

        self.classifier = t.nn.Sequential(
            t.nn.Linear(42, 10)
        )

        self.max_pool = t.nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        """x has shape [m, 1, 28, 28]"""
        x = self.layer1(x)  # [3, 28, 28]
        x = self.max_pool(x)  # [3, 14, 14]
        x = self.layer2(x)  # [8, 10, 10]
        x = self.max_pool(x)  # [8, 5, 5]

        x = x.view(x.size(0), -1)
        x = self.fc(x)  # [42]
        x = self.classifier(x)

        return x