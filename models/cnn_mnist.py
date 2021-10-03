# cnn_mnist from Borna, adapted

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os

from models.ew_layers import EWLinear, EWConv2d


# ew_cnn_mnist
class cnn_mnist(nn.Module):
    """CNN for mnist."""

    def __init__(self, num_classes=10):
        """CNN Builder."""
        super(cnn_mnist, self).__init__()
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            EWConv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 1 to 3
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            EWConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            EWConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            EWConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            EWConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            EWConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            EWLinear(2304, 1024),  # 4x4x256 = 4096 --- 3x3x256 = 2304
            nn.ReLU(inplace=True),
            EWLinear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            EWLinear(512, num_classes)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)  # first dimension batch size

        # fc layer
        x = self.fc_layer(x)

        return x

    # for exponential weighting
    def enable_ew(self, t):
        for layer in self.conv_layer:
            if isinstance(layer, EWConv2d):
                layer.enable(t)

        for layer in self.fc_layer:
            if isinstance(layer, EWLinear):
                layer.enable(t)

    def disable_ew(self):
        for layer in self.conv_layer:
            if isinstance(layer, EWConv2d):
                layer.disable()

        for layer in self.fc_layer:
            if isinstance(layer, EWLinear):
                layer.disable()


# original model
# class cnn_mnist(nn.Module):
#     """CNN."""
#
#     def __init__(self, num_classes=10):
#         """CNN Builder."""
#         super(cnn_mnist, self).__init__()
#         self.conv_layer = nn.Sequential(
#
#             # Conv Layer block 1
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 1 to 3
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # Conv Layer block 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.05),
#
#             # Conv Layer block 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#
#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(2304, 1024),  # 4x4x256 = 4096 --- 3x3x256 = 2304
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.Linear(512, num_classes)
#         )
#
#     def forward(self, x):
#         """Perform forward."""
#
#         # conv layers
#         x = self.conv_layer(x)
#
#         # flatten
#         x = x.view(x.size(0), -1)  # first dimension batch size
#
#         # fc layer
#         x = self.fc_layer(x)
#
#         return x