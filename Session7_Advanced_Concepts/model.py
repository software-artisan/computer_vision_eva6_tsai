'''
Model for CIFAR classification task
'''
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from torchsummary import summary

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False), # RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False), # RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),
        )

        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),
        )

        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),
        )

        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, padding=1, bias=False),
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
            # nn.Dropout2d(0.05),
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)

        x = self.conv2(x) 
        x = self.trans2(x) 

        x = self.conv3(x) 
        x = self.trans3(x)

        x = self.conv4(x)
        x = self.gap(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


# if __name__ == "__main__":
#     pass