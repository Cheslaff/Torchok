# from __future__ import annotations
from typing import Any
import numpy as np

class Add:
    @staticmethod
    def forward(a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items + b.items)

        if a.requires_grad or b.requires_grad:
            out.parents = (a, b)
            out.function = Add()
            out.requires_grad = True
        
        return out
    
    @staticmethod
    def backward(out_grad, a, b) -> tuple:
        grad_a = out_grad
        grad_b = out_grad

        if len(a.shape) == 0:
            grad_a = grad_a.sum()  # Now returns a Tensor
        if len(b.shape) == 0:
            grad_b = grad_b.sum()

        return grad_a, grad_b
    

class Mul:
    @staticmethod
    def forward(a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items * b.items)

        if a.requires_grad or b.requires_grad:
            out.parents = (a, b)
            out.function = Mul()
            out.requires_grad = True
        
        return out
    
    @staticmethod
    def backward(out_grad, a, b) -> tuple:
        grad_a = b * out_grad
        grad_b = a * out_grad

        # Sum gradients if original tensors were scalars
        if len(a.shape) == 0:
            grad_a = grad_a.sum()
        if len(b.shape) == 0:
            grad_b = grad_b.sum()

        return grad_a, grad_b


class Pow:
    @staticmethod
    def forward(a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items ** b.items)
        if a.requires_grad:
            out.parents = (a, b)
            out.function = Pow()
            out.requires_grad = True

        return out
    
    @staticmethod
    def backward(out_grad, a, b) -> tuple:
        grad_a = b * (a ** (b - 1)) * out_grad
        if len(a.shape) == 0:
            grad_a = grad_a.sum()
        return grad_a


class Div:
    @staticmethod
    def forward(a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items / b.items)
        if a.requires_grad or b.requires_grad:
            out.parents = (a, b)
            out.function = Div()
            out.requires_grad = True
        return out

    @staticmethod
    def backward(out_grad, a, b) -> tuple:
        grad_a = out_grad / b
        grad_b = -1 * out_grad * a / (b ** 2)
        
        if len(a.shape) == 0:
            grad_a = grad_a.sum()
        if len(b.shape) == 0:
            grad_b = grad_b.sum()
        return grad_a, grad_b