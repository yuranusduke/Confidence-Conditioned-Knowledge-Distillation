"""
This file contains basic building of AlexNet and AlexNet5 half without dropout
For Cifar10, Note: we do not testify SVHN

Created by Kunhong Yu
Date: 2021/07/16
"""
import torch as t

class AlexNet(t.nn.Module):
    """Define AlexNet model
    Unlike original AlexNet, in order to be compatible with 32 x 32 input, we
    modify conv1 layer to have kernel size of 5 and stride of 1 instead of kernel size
    of 11 and stride of 4, we don't resize 32 x 32 input to be 224 x 224 due to limited
    computation.
    """

    def __init__(self):
        super().__init__()

        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, stride = 1),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(inplace = True)
        )

        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 5, stride = 1, padding = 2),
            t.nn.BatchNorm2d(192),
            t.nn.ReLU(inplace = True)
        )

        self.layer3 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 192, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(384),
            t.nn.ReLU(inplace = True)
        )

        self.layer4 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU(inplace = True)
        )

        self.layer5 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU(inplace = True)
        )

        self.fc = t.nn.Sequential(
            t.nn.Linear(1024, 4096),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(inplace = True)
        )

        self.classifier = t.nn.Sequential(
            t.nn.Linear(4096, 10)
        )

        self.max_pool = t.nn.MaxPool2d(kernel_size = 3, stride = 2)

    def forward(self, x):
        """x has shape [m, 3, 32, 32]"""
        x = self.layer1(x) # [64, 28, 28]
        x = self.max_pool(x) # [64, 13, 13]
        x = self.layer2(x) # [192, 13, 13]
        x = self.max_pool(x) # [192, 6, 6]

        x = self.layer5(self.layer4(self.layer3(x))) # [256, 6, 6]
        x = self.max_pool(x) # [256, 2, 2]

        x = x.view(x.size(0), -1)
        x = self.fc(x) # [4096]
        x = self.classifier(x)

        return x


class AlexNetHalf(t.nn.Module):
    """Define AlexNet half model
    Unlike original AlexNet, in order to be compatible with 32 x 32 input, we
    modify conv1 layer to have kernel size of 5 and stride of 1 instead of kernel size
    of 11 and stride of 4, we don't resize 32 x 32 input to be 224 x 224 due to limited
    computation.
    Note: in the paper, authors did not give details of the net, we just build them from our perspective
    """

    def __init__(self):
        super().__init__()

        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU(inplace = True)
        )

        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 32, out_channels = 96, kernel_size = 5, stride = 1, padding = 2),
            t.nn.BatchNorm2d(96),
            t.nn.ReLU(inplace = True)
        )

        self.layer3 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(192),
            t.nn.ReLU(inplace = True)
        )

        self.layer4 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU(inplace = True)
        )

        self.layer5 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU(inplace = True)
        )

        self.fc = t.nn.Sequential(
            t.nn.Linear(512, 2048),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(2048, 2048),
            t.nn.ReLU(inplace = True)
        )

        self.classifier = t.nn.Sequential(
            t.nn.Linear(2048, 10)
        )

        self.max_pool = t.nn.MaxPool2d(kernel_size = 3, stride = 2)

    def forward(self, x):
        """x has shape [m, 3, 32, 32]"""
        x = self.layer1(x)  # [32, 28, 28]
        x = self.max_pool(x)  # [32, 13, 13]
        x = self.layer2(x)  # [96, 13, 13]
        x = self.max_pool(x)  # [96, 6, 6]

        x = self.layer5(self.layer4(self.layer3(x)))  # [128, 6, 6]
        x = self.max_pool(x)  # [128, 2, 2]

        x = x.view(x.size(0), -1)
        x = self.fc(x)  # [2048]
        x = self.classifier(x)

        return x