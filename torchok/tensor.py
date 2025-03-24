import numpy as np
from typing import Iterable, Callable, Any


class Tensor:
    """torchok.Tensor class supporting arithmetic operations.

    Args:
        items (Iterable): Sequence of values in Tensor

    Attributes:
        items (np.ndarray): np.array() of items sequence.

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
            case _ if isinstance(other, np.ndarray):  # crutch used since np.array aint type
                return Tensor(op(self.items, other))
            case list():
                return Tensor(op(self.items, np.array(other)))
            
        # Scalars section
        return Tensor(op(self.items, other))

    def __add__(self, other: Any) -> 'Tensor':
        return self._match_optypes(other, lambda x, y: x + y)
    
    def __mul__(self, other: Any) -> 'Tensor':
        return self._match_optypes(other, lambda x, y: x * y)
    
    def __radd__(self, other: Any) -> 'Tensor':
        return self + other
    
    def __rmul__(self, other: Any) -> 'Tensor':
        return self * other
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Tensor):
            return np.allclose(self.items, other.items)
        return False

    
    def __repr__(self):
        return f"torchok.Tensor({self.items})"
