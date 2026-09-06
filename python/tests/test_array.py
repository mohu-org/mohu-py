import pytest

import mohu as mu


def test_construction_and_metadata():
    one_d = mu.array([1.0, 2.0, 3.0])
    assert one_d.shape == (3,)
    assert one_d.ndim == 1
    assert one_d.dtype == "float64"
    assert one_d.size == 3
    assert one_d.tolist() == [1.0, 2.0, 3.0]

    two_d = mu.array([[1.0, 2.0], [3.0, 4.0]])
    assert two_d.shape == (2, 2)
    assert two_d.strides == (16, 8)
    assert two_d.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_reshape_and_transpose_preserve_logical_values():
    array = mu.array([[1.0, 2.0], [3.0, 4.0]])
    assert array.reshape((4, 1)).tolist() == [[1.0], [2.0], [3.0], [4.0]]
    assert array.T.shape == (2, 2)
    assert array.T.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert array.transpose().tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert array.reshape((4, 1)).strides == (8, 8)


def test_matmul_operator_and_function():
    lhs = mu.array([[1.0, 2.0], [3.0, 4.0]])
    rhs = mu.array([[5.0, 6.0], [7.0, 8.0]])
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert (lhs @ rhs).tolist() == expected
    assert mu.matmul(lhs, rhs).tolist() == expected


def test_repr_is_readable():
    assert repr(mu.array([[1.0, 2.0], [3.0, 4.0]])) == (
        "mohu.Array([[1.0, 2.0], [3.0, 4.0]], dtype=float64)"
    )


def test_invalid_inputs_raise_python_errors():
    with pytest.raises(ValueError, match="ragged"):
        mu.array([[1.0, 2.0], [3.0]])
    with pytest.raises(TypeError):
        mu.array([[object()]])
    with pytest.raises(ValueError):
        mu.array([[1.0, 2.0]]).reshape((3, 1))
    with pytest.raises(ValueError):
        mu.array([[1.0, 2.0]]) @ mu.array([[1.0, 2.0]])
