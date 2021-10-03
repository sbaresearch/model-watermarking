'''"""from: https://github.com/zhenglisec/Blind-Watermark-for-DNN"""

VGG11/13/16/19 in Pytorch.

Zitate in How To Prove'''

import torch
import torch.nn as nn

from models.ew_layers import EWLinear, EWConv2d


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            EWLinear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            EWLinear(256, num_classes),
        )
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [EWConv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x, track_running_stats=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def freeze_hidden_layers(self):
        self._freeze_layer(self.features)

    def unfreeze_model(self):
        self._freeze_layer(self.features, freeze=False)
        self._freeze_layer(self.classifier, freeze=False)

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

def vgg16(num_classes):
    return VGG('VGG16', num_classes)

