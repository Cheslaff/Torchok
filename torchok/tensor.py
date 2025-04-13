import numpy as np
from typing import List, Tuple, Union
from numpy.typing import ArrayLike


class Tensor:
    """torchok.Tensor class with arithmetic operations support and computation graph tracking.

    Args:
        items (Iterable): Sequence of values in Tensor
        requires_grad (bool): Whether tensor requires gradient computation
        name (str): name for debugging

    Attributes:
        items (np.ndarray): np.array() of items sequence.
        prev (set): set of previous Tensors in computation graph
        requires_grad (bool): whether computation graph tracking is required
        grad (np.array): gradient of current Tensor
        function (class): class with forward and backward methods defined
        name (str): name used for debugging
    
    Properties:
        T: returns transpose of tensor
        shape: returns Tensor shape tuple

    Examples:
        >>> tensor = Tensor([1, 2, 3])
        >>> arr = Tensor([2, 3, 4])
        >>> scalar = 1.4
        >>> (tensor + arr) * scalar  # torchok.Tensor([4.2 7.  9.8])
    """
    def __init__(self, items:ArrayLike, requires_grad:bool=False, name:str=""):
        self.items = np.array(items) if isinstance(items, (list, float, int)) else items
        self.prev = set()
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(items, dtype=np.float64)
        self.function = None
        self.name = name

    def __add__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        from torchok.autogradik.functions import Add
        add = Add()
        return add.forward(self, other)
    
    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        from torchok.autogradik.functions import Mul
        mul = Mul()
        return mul.forward(self, other)
        
    def __pow__(self, other: Union[int, float]) -> 'Tensor':
        from torchok.autogradik.functions import Pow
        pow = Pow()
        return pow.forward(self, other)
    
    def __sub__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        from torchok.autogradik.functions import Sub
        sub = Sub()
        return sub.forward(self, other)
    
    def __truediv__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        from torchok.autogradik.functions import Div
        div = Div()
        return div.forward(self, other)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        from torchok.autogradik.functions import Matmul
        matmul = Matmul()
        return matmul.forward(self, other)
    
    def __getitem__(self, index):
        return Tensor(self.items[index])
    
    def sum(self, dim=None, keepdims=False) -> 'Tensor':
        from torchok.autogradik.functions import Sum
        sum_ = Sum()
        return sum_.forward(self, dim, keepdims)
    
    def mean(self, dim=None, keepdims=False) -> 'Tensor':
        divisor = np.prod(self.items.shape) if dim is None else self.items.shape[dim]
        return self.sum(dim, keepdims) / divisor
    
    def std(self, dim=None, keepdims=False) -> 'Tensor':
        var = ((self - self.mean(dim, keepdims)) ** 2).mean(dim, keepdims)
        return var ** 0.5
    
    def log(self) -> 'Tensor':
        from torchok.autogradik.functions import Log
        log = Log()
        return log.forward(self)
    
    def __radd__(self, other) -> 'Tensor':
        return self + other
    
    def __rmul__(self, other) -> 'Tensor':
        return self * other
    
    def __eq__(self, other: 'Tensor'):
        if isinstance(other, Tensor):
            return np.equal(self.items, other.items)
        return False
    
    def __hash__(self):
        return id(self)
    
    def abs(self):
        from torchok.autogradik.functions import Abs
        abs_ = Abs()
        return abs_.forward(self)
    
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
    
    # Activation Functions
    def relu(self):
        from torchok.autogradik.functions import ReLU
        relu = ReLU()
        return relu.forward(self)
    
    def lrelu(self):
        from torchok.autogradik.functions import LReLU
        lrelu = LReLU()
        return lrelu.forward(self)
    
    def sigmoid(self):
        from torchok.autogradik.functions import Sigmoid
        sigmoid = Sigmoid()
        return sigmoid.forward(self)
    
    def tanh(self):
        from torchok.autogradik.functions import Tanh
        tanh = Tanh()
        return tanh.forward(self)
    
    def softmax(self):
        from torchok.autogradik.functions import Softmax
        softmax = Softmax()
        return softmax.forward(self)

    # Properties
    @property
    def shape(self) -> Tuple:
        return self.items.shape

    @property
    def T(self) -> 'Tensor':
        return Tensor(self.items.T)

    def __repr__(self):
        return f"torchok.Tensor({self.items})"


# Functions for Tensor generation
def zeros(*shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad=requires_grad)

def ones(*shape, requires_grad=False):
    return Tensor(np.ones(shape))

def randn(*shape, requires_grad=False):
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

def arange(start=0, stop=None, step=1, requires_grad=False):
    return Tensor(np.arange(start, stop, step=step), requires_grad=requires_grad)

def arange(start=0, stop=None, num=1, requires_grad=False):
    return Tensor(np.linspace(start, stop, num=num), requires_grad=requires_grad)
