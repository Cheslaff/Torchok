import numpy as np
from torchok.tensor import Tensor
from hypothesis import given
from hypothesis.strategies import integers, floats, lists, composite

@composite
def create_tensor(draw):
    rows = draw(integers(1, 5))
    cols = draw(integers(1, 5))

    return draw(lists(lists(floats(-10, 10), min_size=cols, max_size=cols),
                      min_size=rows, max_size=rows))



@given(create_tensor())
def test_matmul_on_transpose(matrix):
    tensor = Tensor(matrix)
    tensor_t = tensor.T

    numpy_arr = np.array(matrix)
    numpy_arr_t = numpy_arr.T

    assert np.allclose((tensor @ tensor_t).items, numpy_arr @ numpy_arr_t)
