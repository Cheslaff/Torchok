# from __future__ import annotations
from typing import Any


class Add:
    def forward(self, a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items + b.items)
        
        self.a = a.items
        self.b = b.items

        if a.requires_grad or b.requires_grad:
            out.parents = (a, b)
            out.function = self
        
        return out
    
    def backward(self, out_grad) -> tuple:
        grad_a = out_grad
        grad_b = out_grad
        return grad_a, grad_b
    

class Mul:
    def forward(self, a, b) -> 'Tensor':
        from torchok.tensor import Tensor
        out = Tensor(a.items * b.items)
        
        self.a = a.items
        self.b = b.items

        if a.requires_grad or b.requires_grad:
            out.parents = (a, b)
            out.function = self
        
        return out
    
    def backward(self, out_grad) -> tuple:
        grad_a = self.a * out_grad
        grad_b = self.b * out_grad
        return grad_a, grad_b
    