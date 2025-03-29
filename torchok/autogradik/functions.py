# from __future__ import annotations
from typing import Any


class Add:
    @staticmethod
    def forward(a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items + b.items)

        if a.requires_grad or b.requires_grad:
            out.parents = (a, b)
            out.function = Add()
        
        return out
    
    @staticmethod
    def backward(out_grad, a, b) -> tuple:
        grad_a = out_grad
        grad_b = out_grad
        return grad_a, grad_b
    

class Mul:
    @staticmethod
    def forward(a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items * b.items)

        if a.requires_grad or b.requires_grad:
            out.parents = (a, b)
            out.function = Mul()
        
        return out
    
    @staticmethod
    def backward(out_grad, a, b) -> tuple:
        grad_a = b * out_grad
        grad_b = a * out_grad

        # Sum gradients if original tensors were scalars
        if len(a.shape) == 0:
            grad_a = grad_a.items.sum()
        if len(b.shape) == 0:
            grad_b = grad_b.items.sum()

        return grad_a, grad_b
