

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['densenet']


from torch.autograd import Variable

from models.ew_layers import EWLinear, EWConv2d


# original from: https://gist.github.com/koshian2/f1ecf57390d5efe24f6d67f3e596b43b
# adapted with EW Layers

class DenseBlock(nn.Module):
    def __init__(self, input_channels, growth_rate):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels + growth_rate
        # Layers
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = EWConv2d(input_channels, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = EWConv2d(128, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

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

class TransitionBlock(nn.Module):
    def __init__(self, input_channels, compression):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = int(input_channels * compression)
        # Layers
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = EWConv2d(input_channels, self.output_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return nn.AvgPool2d(kernel_size=2)(out)

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()

class DenseNet(nn.Module):
    def __init__(self, num_classes=10, growth_rate=16, compression_factor=0.5, blocks=[1, 2, 4, 3]):
        super().__init__()
        self.num_classes = num_classes
        # 成長率(growth_rate)：DenseBlockで増やすフィルターの数
        self.k = growth_rate
        # 圧縮率(compression_factor)：Transitionレイヤーで圧縮するフィルターの比
        self.compression = compression_factor
        # ブロック構成
        self.blocks = blocks
        # 履歴
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "time": []}
        # モデル作成
        self.make_model()

    def make_dense_block(self, input_channels, nb_blocks):
        n_channels = input_channels
        layers = []
        for i in range(nb_blocks):
            item = DenseBlock(n_channels, self.k)
            layers.append(item)
            n_channels = item.output_channels
        return nn.Sequential(*layers), n_channels

    def make_transition_block(self, input_channels):
        item = TransitionBlock(input_channels, self.compression)
        return item, item.output_channels

    def make_model(self):
        # blocks=[6,12,24,16]とするとDenseNet-121の設定に準じる
        # 端数を出さないようにフィルター数16にする
        n = 16
        self.conv1 = EWConv2d(3, n, kernel_size=1)
        # DenseBlock - TransitionLayer - DenseBlock…
        self.dense1, n = self.make_dense_block(n, self.blocks[0])
        self.trans1, n = self.make_transition_block(n)
        self.dense2, n = self.make_dense_block(n, self.blocks[1])
        self.trans2, n = self.make_transition_block(n)
        self.dense3, n = self.make_dense_block(n, self.blocks[2])
        self.trans3, n = self.make_transition_block(n)
        self.dense4, n = self.make_dense_block(n, self.blocks[3])
        self.gap = nn.AvgPool2d(kernel_size=4) # 最後は(4,4)
        self.fc = EWLinear(n, self.num_classes) # softmaxは損失関数で
        self.gap_channels = n

    # モデルの作成
    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        out = self.dense4(out)
        out = self.gap(out)
        out = out.view(-1, self.gap_channels)
        out = self.fc(out)
        return out

    def freeze_hidden_layers(self):
        self._freeze_layer(self.conv1)
        self._freeze_layer(self.dense1)
        self._freeze_layer(self.trans1)
        self._freeze_layer(self.dense2)
        self._freeze_layer(self.trans2)
        self._freeze_layer(self.dense3)
        self._freeze_layer(self.trans3)
        self._freeze_layer(self.dense4)


    def unfreeze_model(self):
        self._freeze_layer(self.conv1, freeze=False)
        self._freeze_layer(self.dense1, freeze=False)
        self._freeze_layer(self.trans1, freeze=False)
        self._freeze_layer(self.dense2, freeze=False)
        self._freeze_layer(self.trans2, freeze=False)
        self._freeze_layer(self.dense3, freeze=False)
        self._freeze_layer(self.trans3, freeze=False)
        self._freeze_layer(self.dense4, freeze=False)
        self._freeze_layer(self.fc, freeze=False)

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

def densenet(num_classes=10):
    """
    Constructs a DenseNet-121 with growth rate 16.
    """
    return DenseNet(num_classes=num_classes, growth_rate=16, blocks=[6, 12, 24, 16])



# original from Borna, adapted

# class Bottleneck(nn.Module):
#     def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
#         super(Bottleneck, self).__init__()
#         planes = expansion * growthRate
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = EWConv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = EWConv2d(planes, growthRate, kernel_size=3,
#                                padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropRate = dropRate
#
#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.dropRate > 0:
#             out = F.dropout(out, p=self.dropRate, training=self.training)
#
#         out = torch.cat((x, out), 1)
#
#         return out
#
#     # for exponential weighting
#     def enable_ew(self, t):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.enable(t)
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.enable(t)
#
#     def disable_ew(self):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.disable()
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.disable()
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
#         super(BasicBlock, self).__init__()
#         planes = expansion * growthRate
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = EWConv2d(inplanes, growthRate, kernel_size=3,
#                                padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropRate = dropRate
#
#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         if self.dropRate > 0:
#             out = F.dropout(out, p=self.dropRate, training=self.training)
#
#         out = torch.cat((x, out), 1)
#
#         return out
#
#     # for exponential weighting
#     def enable_ew(self, t):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.enable(t)
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.enable(t)
#
#     def disable_ew(self):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.disable()
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.disable()
#
#
# class Transition(nn.Module):
#     def __init__(self, inplanes, outplanes):
#         super(Transition, self).__init__()
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = EWConv2d(inplanes, outplanes, kernel_size=1,
#                                bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = F.avg_pool2d(out, 2)
#         return out
#
#     # for exponential weighting
#     def enable_ew(self, t):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.enable(t)
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.enable(t)
#
#     def disable_ew(self):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.disable()
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.disable()
#
# class DenseNet(nn.Module):
#
#     def __init__(self, num_classes=10, depth=22, block=Bottleneck,
#         dropRate=0, growthRate=12, compressionRate=2):
#         super(DenseNet, self).__init__()
#
#         assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
#         n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
#
#         self.growthRate = growthRate
#         self.dropRate = dropRate
#
#         # self.inplanes is a global variable used across multiple
#         # helper functions
#         self.inplanes = growthRate * 2
#         self.conv1 = EWConv2d(3, self.inplanes, kernel_size=3, padding=1,
#                                bias=False)
#         self.dense1 = self._make_denseblock(block, n)
#         self.trans1 = self._make_transition(compressionRate)
#         self.dense2 = self._make_denseblock(block, n)
#         self.trans2 = self._make_transition(compressionRate)
#         self.dense3 = self._make_denseblock(block, n)
#         self.bn = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(8)
#         self.fc = EWLinear(self.inplanes, num_classes)
#
#         # Weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, EWConv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_denseblock(self, block, blocks):
#         layers = []
#         for i in range(blocks):
#             # Currently we fix the expansion ratio as the default value
#             layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
#             self.inplanes += self.growthRate
#
#         return nn.Sequential(*layers)
#
#     def _make_transition(self, compressionRate):
#         inplanes = self.inplanes
#         outplanes = int(math.floor(self.inplanes // compressionRate))
#         self.inplanes = outplanes
#         return Transition(inplanes, outplanes)
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#
#         x = self.trans1(self.dense1(x))
#         x = self.trans2(self.dense2(x))
#         x = self.dense3(x)
#         x = self.bn(x)
#         x = self.relu(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#     # for exponential weighting
#     def enable_ew(self, t):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.enable(t)
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.enable(t)
#
#     def disable_ew(self):
#         for layer in self.conv_layer:
#             if isinstance(layer, EWConv2d):
#                 layer.disable()
#
#         for layer in self.fc_layer:
#             if isinstance(layer, EWLinear):
#                 layer.disable()
#
#
# def densenet(**kwargs):
#     """
#     Constructs a ResNet model.
#     """
#     return DenseNet(**kwargs)


# original model
# class Bottleneck(nn.Module):
#     def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
#         super(Bottleneck, self).__init__()
#         planes = expansion * growthRate
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
#                                padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropRate = dropRate
#
#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.dropRate > 0:
#             out = F.dropout(out, p=self.dropRate, training=self.training)
#
#         out = torch.cat((x, out), 1)
#
#         return out
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
#         super(BasicBlock, self).__init__()
#         planes = expansion * growthRate
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,
#                                padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropRate = dropRate
#
#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         if self.dropRate > 0:
#             out = F.dropout(out, p=self.dropRate, training=self.training)
#
#         out = torch.cat((x, out), 1)
#
#         return out
#
#
# class Transition(nn.Module):
#     def __init__(self, inplanes, outplanes):
#         super(Transition, self).__init__()
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
#                                bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = F.avg_pool2d(out, 2)
#         return out
#
#
# class DenseNet(nn.Module):
#
#     def __init__(self, num_classes=10, depth=22, block=Bottleneck,
#         dropRate=0, growthRate=12, compressionRate=2):
#         super(DenseNet, self).__init__()
#
#         assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
#         n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
#
#         self.growthRate = growthRate
#         self.dropRate = dropRate
#
#         # self.inplanes is a global variable used across multiple
#         # helper functions
#         self.inplanes = growthRate * 2
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
#                                bias=False)
#         self.dense1 = self._make_denseblock(block, n)
#         self.trans1 = self._make_transition(compressionRate)
#         self.dense2 = self._make_denseblock(block, n)
#         self.trans2 = self._make_transition(compressionRate)
#         self.dense3 = self._make_denseblock(block, n)
#         self.bn = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(self.inplanes, num_classes)
#
#         # Weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_denseblock(self, block, blocks):
#         layers = []
#         for i in range(blocks):
#             # Currently we fix the expansion ratio as the default value
#             layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
#             self.inplanes += self.growthRate
#
#         return nn.Sequential(*layers)
#
#     def _make_transition(self, compressionRate):
#         inplanes = self.inplanes
#         outplanes = int(math.floor(self.inplanes // compressionRate))
#         self.inplanes = outplanes
#         return Transition(inplanes, outplanes)
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#
#         x = self.trans1(self.dense1(x))
#         x = self.trans2(self.dense2(x))
#         x = self.dense3(x)
#         x = self.bn(x)
#         x = self.relu(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#
# def densenet(**kwargs):
#     """
#     Constructs a ResNet model.
#     """
#     return DenseNet(**kwargs)
