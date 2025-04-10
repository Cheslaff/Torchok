import numpy as np
from typing import Iterable, Callable, Any, Tuple


class Tensor:
    """torchok.Tensor class supporting arithmetic operations.

    Args:
        items (Iterable): Sequence of values in Tensor

    Attributes:
        items (np.ndarray): np.array() of items sequence.
    
    Properties:
        T: returns transpose of tensor
        shape: returns Tensor shape tuple

    Examples:
        >>> tensor = Tensor([1, 2, 3])
        >>> arr = [2, 3, 4]
        >>> scalar = 1.4
        >>> (tensor + arr) * scalar  # torchok.Tensor([4.2 7.  9.8])
    """
    def __init__(self, items: Iterable) -> None:
        self.items = items if isinstance(items, np.ndarray) else np.array(items)

    def _match_optypes(self, other: Any, op: Callable) -> 'Tensor':
        # Iterables section
        match other:
            case Tensor():
                return Tensor(op(self.items, other.items))
            case _ if isinstance(other, np.ndarray):  # np.array aint type
                return Tensor(op(self.items, other))
            case list() | tuple():
                return Tensor(op(self.items, np.array(other)))
            
        # Scalars section
        return Tensor(op(self.items, other))

    def __add__(self, other: Any) -> 'Tensor':
        return self._match_optypes(other, lambda x, y: x + y)
    
    def __mul__(self, other: Any) -> 'Tensor':
        return self._match_optypes(other, lambda x, y: x * y)
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Tensor):
            return np.allclose(self.items, other.items)
        return False

    def __matmul__(self, other: Any) -> 'Tensor':
        try:
            return self._match_optypes(other, lambda x, y: x @ y)
        except ValueError:
            raise ValueError("Shape mismatch error")
        
    def __getitem__(self, index: int) -> 'Tensor':
        return Tensor(self.items[index])

    def __radd__(self, other: Any) -> 'Tensor':
        return self + other

    def __rmul__(self, other: Any) -> 'Tensor':
        return self * other
    
    def __rmatmul__(self, other: Any) -> 'Tensor':
        return self @ other
    
    @property
    def T(self) -> 'Tensor':
        return Tensor(self.items.T)
    
    @property
    def shape(self) -> Tuple:
        return self.items.shape

    def __repr__(self):
        return f"torchok.Tensor({self.items})"
