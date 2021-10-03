''' Exponential weighting convolutional and dense layers (linear layer) '''

import torch
import torch.nn as nn

import math

from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import  _pair
from torch.nn import functional as F, Linear, init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from torch.nn.common_types import _size_2_t


class EWConv2d(_ConvNd):
    r"""Adapted the forward method from Conv2d, included two new attributes (t, is_ew_enabled) and two methods (enable, disable).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(EWConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.t = 1  # temperature
        self.is_ew_enabled = False

    def enable(self, t):
        self.is_ew_enabled = True  # enable exponential weighting
        self.t = t  # set temperature

    def disable(self):
        self.is_ew_enabled = False

    def ew(self, theta):
        exp = torch.exp(torch.abs(theta) * self.t)
        numerator = exp
        denominator = torch.max(exp)
        return torch.mul(numerator / denominator, theta)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        if self.is_ew_enabled:
            return self._conv_forward(input, self.ew(self.weight))
        else:
            return self._conv_forward(input, self.weight)


class EWLinear(Module):
    r"""Adapted the forward method from Linear, included two new attributes (t, is_ew_enabled) and two methods (enable, disable).
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(EWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.t = 1  # temperature
        self.is_ew_enabled = False

    def enable(self, t):
        self.is_ew_enabled = True  # enable exponential weighting
        self.t = t  # set temperature

    def disable(self):
        self.is_ew_enabled = False

    def ew(self, theta):
        exp = torch.exp(torch.abs(theta) * self.t)
        numerator = exp
        denominator = torch.max(exp)
        return torch.mul(numerator / denominator, theta)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.is_ew_enabled:
            return F.linear(input, self.ew(self.weight), self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
