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

class BatchNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torchok.ones(dim)
        self.beta = torchok.zeros(dim)
        self.running_mean = torchok.zeros(dim)
        self.running_std = torchok.ones(dim)

    def forward(self, x):
        if self.training:
            mean_x = x.mean(0, keepdims=True)
            std_x = x.std(0, keepdims=True)
        else:
            mean_x = self.running_mean
            std_x = self.running_std
        x_hat = (x - mean_x) / (std_x + self.eps)
        self.out = self.gamma * x_hat + self.beta
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std_x
        return self.out


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
    