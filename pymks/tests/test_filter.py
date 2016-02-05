import numpy as np


def test_frequency_2_real_and_back():
    from pymks.filter import Filter
    from pymks import PrimitiveBasis

    X = np.zeros((1, 3, 2, 2))
    X[0, 0, 0] = np.arange(1, 3) * 9
    p_basis = PrimitiveBasis(2)
    p_basis._axes = (1, 2)
    p_basis._axes_shape = (3, 3)
    X_result = np.ones((1, 3, 3, 2)) * np.arange(1, 3)[None, None, None, :]
    filter_ = Filter(X, p_basis)
    assert np.allclose(filter_._frequency_2_real(copy=True), X_result)
    assert np.allclose(filter_._real_2_frequency(X_result), X)

if __name__ == '__main__':
    test_frequency_2_real_and_back()
