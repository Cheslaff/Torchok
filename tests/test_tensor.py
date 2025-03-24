from torchok.tensor import Tensor
from hypothesis import given, settings, example
from hypothesis.strategies import floats, lists, integers, one_of
import numpy as np
import pytest
from hypothesis.strategies import composite


"""
LLM generated stuff down there.
Not familiar with hypothesis yet (gotta fix it)
Anyways, temporary tests to break free from this shit
"""
@composite
def tensor_and_compatible_other(draw):
    shape = draw(lists(integers(min_value=1, max_value=5), min_size=1, max_size=3))
    
    tensor_elements = draw(lists(
        floats(allow_nan=False, allow_infinity=False), 
        min_size=shape[0], 
        max_size=shape[0]
    ))
    tensor = Tensor(tensor_elements)
    
    other = draw(one_of(
        floats(allow_nan=False, allow_infinity=False),
        lists(floats(allow_nan=False, allow_infinity=False),
              min_size=shape[0], 
              max_size=shape[0]).map(Tensor)
    ))
    
    return tensor, other

@settings(max_examples=100)
@given(tensor_and_compatible_other())
@example((Tensor([0.0]), 0.0))
@example((Tensor([1.0, -1.0]), 1.0))
@example((Tensor([2.0]), Tensor([3.0])))
def test_addition(pair):
    tensor, other = pair
    result = tensor + other
    
    if isinstance(other, Tensor):
        expected = Tensor(tensor.items + other.items)
    else:
        expected = Tensor(tensor.items + other)
    
    assert result.items.shape == expected.items.shape, (
        f"Shape mismatch: {result.items.shape} vs {expected.items.shape}"
    )
    assert np.allclose(result.items, expected.items, atol=1e-8), (
        f"Value mismatch:\nResult: {result.items}\nExpected: {expected.items}"
    )

