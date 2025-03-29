from torchok.tensor import Tensor
import numpy as np


w = Tensor([1, 2, 3], requires_grad=True)
x = Tensor([2, 3, 4], requires_grad=True)
b = Tensor(7, requires_grad=True)

z = w * x + b
z.backward()
