"""from: https://github.com/zhenglisec/Blind-Watermark-for-DNN

Zitate in How To Prove"""

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

from models.ew_layers import EWLinear, EWConv2d

class LeNet1(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet1, self).__init__()
        self.conv1 = EWConv2d(1, 4, kernel_size=5, stride=1, padding=2)
        # self.conv1 = EWConv2d(1,4, kernel_size=5, stride=1, padding=0)
        self.conv2 = EWConv2d(4, 12, kernel_size=5, stride=1, padding=2)
        # self.conv1 = EWConv2d(4,12, kernel_size=5, stride=1, padding=0)
        self.fc1 = EWLinear(588, num_classes)
        # self.fc1 = EWLinear(_, num_classes)

    def forward(self, x, index=-1, metric=0):
        layer = F.relu(self.conv1(x))
        layer = F.max_pool2d(layer, 2)
        #layer = F.avg_pool2d(layer, 2, padding=0, stride=2)
        layer = F.relu(self.conv2(layer))
        layer = F.max_pool2d(layer, 2)
        # layer = F.avg_pool2d(layer, 2, padding=0, stride=2)

        layer = layer.view(-1, 588)
        layer = self.fc1(layer)
        output = F.log_softmax(layer, dim=1)
        return output

    def freeze_hidden_layers(self):
        self._freeze_layer(self.conv1)
        self._freeze_layer(self.conv2)

    def unfreeze_model(self):
        self._freeze_layer(self.conv1, freeze=False)
        self._freeze_layer(self.conv2, freeze=False)
        self._freeze_layer(self.fc1, freeze=False)

    def _freeze_layer(self, layer, freeze=True):
        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()

class LeNet3(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet3, self).__init__()
        self.conv1 = EWConv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # self.conv1 = EWConv2d(1, 4, kernel_size=5, stride=1, padding=0)
        self.conv2 = EWConv2d(6, 16, kernel_size=5, stride=1, padding=2)
        # self.conv2 = EWConv2d(4, 16, kernel_size=5, stride=1, padding=0)

        self.fc1 = EWLinear(784, 84)
        # self.fc1 = EWLinear(_, 120)
        # self.fc1 = EWLinear(120, num_classes)
        self.fc2 = EWLinear(84, num_classes)

    def forward(self, x):
        layer0 = F.relu(self.conv1(x))
        layer1 = F.max_pool2d(layer0, 2)
        # layer1 = F.avg_pool2d(layer0, 2, padding=0, stride=2)
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.max_pool2d(layer2, 2)
        # layer3 = F.avg_pool2d(layer2, 2, padding=0, stride=2)

        layer_ = layer3.view(-1, 784)
        layer4 = F.relu(self.fc1(layer_))
        layer5 = self.fc2(layer4)
        output = F.log_softmax(layer5, dim=1)
        return output

    def freeze_hidden_layers(self):
        self._freeze_layer(self.conv1)
        self._freeze_layer(self.conv2)

    def unfreeze_model(self):
        self._freeze_layer(self.conv1, freeze=False)
        self._freeze_layer(self.conv2, freeze=False)
        self._freeze_layer(self.fc1, freeze=False)
        self._freeze_layer(self.fc2, freeze=False)

    def _freeze_layer(self, layer, freeze=True):
        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv_layer = nn.Sequential(
            EWConv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            EWConv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            EWLinear(784, 120),
            nn.ReLU(inplace=True),
            EWLinear(120, 84),
            nn.ReLU(inplace=True),
            EWLinear(84, num_classes)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)  # first dimension batch size

        # fc layer
        x = self.fc_layer(x)

        return F.log_softmax(x, dim=1)

    def freeze_hidden_layers(self):
        self._freeze_layer(self.conv_layer[0])
        self._freeze_layer(self.conv_layer[3])

    def unfreeze_model(self):
        self._freeze_layer(self.conv_layer[0], freeze=False)
        self._freeze_layer(self.conv_layer[3], freeze=False)
        self._freeze_layer(self.fc_layer[0], freeze=False)
        self._freeze_layer(self.fc_layer[2], freeze=False)
        self._freeze_layer(self.fc_layer[4], freeze=False)

    def _freeze_layer(self, layer, freeze=True):
        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()



class LeNet5_old(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_old, self).__init__()
        self.conv1 = EWConv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # self.conv1 = EWConv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = EWConv2d(6, 16, kernel_size=5, stride=1, padding=2)
        # self.conv2 = EWConv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = EWLinear(784, 120)
        # self.fc1 = EWLinear(_, 140)
        self.fc2 = EWLinear(120, 84)
        # self.fc2 = EWLinear(140, 84)
        self.fc3 = EWLinear(84, num_classes)

    def forward(self, x):
        layer0 = F.relu(self.conv1(x))
        layer1 = F.max_pool2d(layer0, 2)
        # layer1 = F.avg_pool2d(layer0, 2, padding=0, stride=2)
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.max_pool2d(layer2, 2)
        # layer3 = F.max_pool2d(layer2, 2, padding=0, stride=2)

        layer_ = layer3.view(-1, 784)
        layer4 = F.relu(self.fc1(layer_))
        layer5 = F.relu(self.fc2(layer4))
        layer6 = self.fc3(layer5)
        output = F.log_softmax(layer6, dim=1)

        return output

    def freeze_hidden_layers(self):
        self._freeze_layer(self.conv1)
        self._freeze_layer(self.conv2)

    def unfreeze_model(self):
        self._freeze_layer(self.conv1, freeze=False)
        self._freeze_layer(self.conv2, freeze=False)
        self._freeze_layer(self.fc1, freeze=False)
        self._freeze_layer(self.fc2, freeze=False)
        self._freeze_layer(self.fc3, freeze=False)

    def _freeze_layer(self, layer, freeze=True):
        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()


def lenet1(**kwargs):
    return LeNet1(**kwargs)


def lenet3(**kwargs):
    return LeNet3(**kwargs)


def lenet5(**kwargs):
    return LeNet5(**kwargs)
