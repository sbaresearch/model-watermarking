# original from: https://github.com/Coderx7/SimpleNet_Pytorch/blob/master/models/simplenet.py
'''
SimplerNetV1 in Pytorch.

The implementation is basded on :
https://github.com/D-X-Y/ResNeXt-DenseNet
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter

from models.ew_layers import EWLinear, EWConv2d


class simplenet_mnist(nn.Module):
    # for MNIST dataset
    def __init__(self, num_classes=10, simpnet_name='simplenet_mnist'):
        super(simplenet_mnist, self).__init__()
        self.features = self._make_layers()
        self.glob_dropout = nn.Dropout2d(p=0.1)
        self.classifier = EWLinear(256, num_classes)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("STATE_DICT: {}".format(name))
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
                    name, own_state[name].size(), param.size()))

    def forward(self, x):
        out = self.features(x)

        # Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = self.glob_dropout(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()

    def _make_layers(self):

        model = nn.Sequential(
            EWConv2d(1, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            # nn.Dropout2d(p=0.1),

            EWConv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

        )

        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, EWConv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model

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


class simplenet(nn.Module):
    # can be used with CIFAR-10 and CIFAR-100
    # probably won't be used
    def __init__(self, num_classes=10):
        super(simplenet, self).__init__()
        # print(simpnet_name)
        self.features = self._make_layers()  # self._make_layers(cfg[simpnet_name])
        self.classifier = EWLinear(256, num_classes)
        self.drp = nn.Dropout(0.1)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        # print(own_state.keys())
        # for name, val in own_state:
        # print(name)

        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("STATE_DICT: {}".format(name))
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
                    name, own_state[name].size(), param.size()))

    def forward(self, x):
        out = self.features(x)

        # Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        # out = F.dropout2d(out, 0.1, training=True)
        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()

    def _make_layers(self):

        model = nn.Sequential(
            EWConv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            EWConv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            EWConv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

        )

        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, EWConv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model

# original model:
# class simplenet_mnist(nn.Module):
#     # for MNIST dataset
#     def __init__(self, num_classes=10, simpnet_name='simplenet_mnist'):
#         super(simplenet_mnist, self).__init__()
#         self.features = self._make_layers()
#         self.glob_dropout = nn.Dropout2d(p=0.1)
#         self.classifier = nn.Linear(256, num_classes)
#
#     def load_my_state_dict(self, state_dict):
#
#         own_state = self.state_dict()
#
#         # print(own_state.keys())
#         # for name, val in own_state:
#         # print(name)
#         for name, param in state_dict.items():
#             name = name.replace('module.', '')
#             if name not in own_state:
#                 # print(name)
#                 continue
#             if isinstance(param, nn.Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#             print("STATE_DICT: {}".format(name))
#             try:
#                 own_state[name].copy_(param)
#             except:
#                 print('While copying the parameter named {}, whose dimensions in the model are'
#                       ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
#                     name, own_state[name].size(), param.size()))
#
#     def forward(self, x):
#         #print(x.size())
#         out = self.features(x)
#
#         #Global Max Pooling
#         out = F.max_pool2d(out, kernel_size=out.size()[2:])
#         out = self.glob_dropout(out)
#
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out
#
#     def _make_layers(self):
#
#         model = nn.Sequential(
#                              nn.Conv2d(1, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#                              nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#                              nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#                              nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#
#                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#                              nn.Dropout2d(p=0.1),
#
#
#                              nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#                              nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#                              nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#
#
#                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#                              nn.Dropout2d(p=0.1),
#
#
#                              nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#
#                              nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#
#
#                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#                              nn.Dropout2d(p=0.1),
#
#
#
#                              nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#
#
#                              #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#                              #nn.Dropout2d(p=0.1),
#
#
#                              nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
#                              nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#
#
#                              nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
#                              nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#
#                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#                              nn.Dropout2d(p=0.1),
#
#
#                              nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#                              nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#                              nn.ReLU(inplace=True),
#
#                             )
#
#         for m in model.modules():
#           if isinstance(m, nn.Conv2d):
#             nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
#
#         return model
#
#
# class simplenet(nn.Module):
#     # can be used with CIFAR-10 and CIFAR-100
#     # probably won't be used
#     def __init__(self, num_classes=10):
#         super(simplenet, self).__init__()
#         # print(simpnet_name)
#         self.features = self._make_layers()  # self._make_layers(cfg[simpnet_name])
#         self.classifier = nn.Linear(256, num_classes)
#         self.drp = nn.Dropout(0.1)
#
#     def load_my_state_dict(self, state_dict):
#
#         own_state = self.state_dict()
#
#         # print(own_state.keys())
#         # for name, val in own_state:
#         # print(name)
#
#         for name, param in state_dict.items():
#             name = name.replace('module.', '')
#             if name not in own_state:
#                 # print(name)
#                 continue
#             if isinstance(param, Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#             print("STATE_DICT: {}".format(name))
#             try:
#                 own_state[name].copy_(param)
#             except:
#                 print('While copying the parameter named {}, whose dimensions in the model are'
#                       ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
#                     name, own_state[name].size(), param.size()))
#
#     def forward(self, x):
#         out = self.features(x)
#
#         # Global Max Pooling
#         out = F.max_pool2d(out, kernel_size=out.size()[2:])
#         # out = F.dropout2d(out, 0.1, training=True)
#         out = self.drp(out)
#
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out
#
#     def _make_layers(self):
#
#         model = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#             nn.Dropout2d(p=0.1),
#
#             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#             nn.Dropout2d(p=0.1),
#
#             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#             nn.Dropout2d(p=0.1),
#
#             nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#             nn.Dropout2d(p=0.1),
#
#             nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
#             nn.Dropout2d(p=0.1),
#
#             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
#             nn.ReLU(inplace=True),
#
#         )
#
#         for m in model.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
#
#         return model
