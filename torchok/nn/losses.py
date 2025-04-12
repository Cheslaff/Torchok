import numpy as np
import torchok
from torchok import Tensor
from torchok import nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, pred, y):
        loss  = (pred  -  y) **  2
        return loss
