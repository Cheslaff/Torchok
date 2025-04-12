import numpy as np
import torchok
from torchok import Tensor
from torchok import nn


class Linear(nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.W = torchok.randn(fan_in, fan_out, requires_grad=True) / (fan_in)**0.5
        self.b = torchok.randn(fan_out, requires_grad=True) * 0.1 if bias else None

    def forward(self, x):
        out = x @ self.W
        if self.b is not None:
            out += self.b
        return out


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        from torchok.autogradik.functions import ReLU as act
        self.relu = act()

    def forward(self, x):
        out = self.relu.forward(x)
        return out
    

class LReLU(nn.Module):
    def __init__(self):
        super().__init__()
        from torchok.autogradik.functions import LReLU as act
        self.lrelu = act()

    def forward(self, x):
        out = self.lrelu.forward(x)
        return out
    

class Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        from torchok.autogradik.functions import Tanh as act
        self.tanh = act()

    def forward(self, x):
        out = self.tanh.forward(x)
        return out


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        from torchok.autogradik.functions import Sigmoid as act
        self.sigmoid = act()

    def forward(self, x):
        out = self.sigmoid.forward(x)
        return out


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        from torchok.autogradik.functions import Softmax as act
        self.softmax = act()

    def forward(self, x):
        out = self.softmax.forward(x)
        return out
    