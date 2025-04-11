import numpy as np
from typing import List, Callable, Tuple


class Tensor:
    def __init__(self, items:List, requires_grad=False, name=""):
        self.items = np.array(items) if isinstance(items, list) else items
        self.prev = set()
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(items, dtype=np.float64)
        self.function = None
        self.name = name

    def __add__(self, other) -> 'Tensor':
        from autogradik.functions import Add
        add = Add()
        return add.forward(self, other)
    
    def __mul__(self, other) -> 'Tensor':
        from autogradik.functions import Mul
        mul = Mul()
        return mul.forward(self, other)
        
    def __pow__(self, other: int | float) -> 'Tensor':
        from autogradik.functions import Pow
        pow = Pow()
        return pow.forward(self, other)
    
    def __sub__(self, other) -> 'Tensor':
        from autogradik.functions import Sub
        sub = Sub()
        return sub.forward(self, other)
    
    def __truediv__(self, other) -> 'Tensor':
        from autogradik.functions import Div
        div = Div()
        return div.forward(self, other)
    
    def __matmul__(self, other) -> 'Tensor':
        from autogradik.functions import Matmul
        matmul = Matmul()
        return matmul.forward(self, other)
    
    def sum(self) -> 'Tensor':
        from autogradik.functions import Sum
        return Sum().forward(self)
    
    def __radd__(self, other) -> 'Tensor':
        return self + other
    
    def __rmul__(self, other) -> 'Tensor':
        return self * other
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return np.equal(self.items, other.items)
        return False
    
    def __hash__(self):
        return id(self)
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited and v.requires_grad:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.items, dtype=np.float64)

        # stdout = [v.name for v in reversed(topo)]
        # print(f"TOPO: {stdout}; {[v.function for v in reversed(topo)]}")
        for v in reversed(topo):
            if v.function is None:
                continue
            v.function.backward()
            

    @property
    def shape(self) -> Tuple:
        return self.items.shape

    @property
    def T(self) -> 'Tensor':
        return Tensor(self.items.T)

    def __repr__(self):
        return f"torchok.Tensor({self.items})"
