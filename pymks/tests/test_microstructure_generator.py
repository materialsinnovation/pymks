import pytest
import numpy as np
from pymks.datasets import make_microstructure


def test_size_and_grain_size_failure():
    with pytest.raises(RuntimeError):
        make_microstructure(n_samples=1, size=(7, 7), grain_size=(8, 1))


def test_volume_fraction_failure():
    with pytest.raises(RuntimeError):
        make_microstructure(n_samples=1, volume_fraction=(0.3, 0.6))


def test_volume_fraction_with_n_phases_failure():
    with pytest.raises(RuntimeError):
        make_microstructure(n_samples=1, size=(7, 7), n_phases=3,
                            volume_fraction=(0.5, 0.5))


def test_percent_variance_exceeds_limit_failure():
    with pytest.raises(RuntimeError):
        make_microstructure(n_samples=1, size=(7, 7), n_phases=3,
                            volume_fraction=(0.3, 0.3, 0.4), percent_variance=0.5)


def test_volume_fraction():
    X = make_microstructure(n_samples=1, n_phases=3,
                            volume_fraction=(0.3, 0.2, 0.5))
    assert np.allclose(np.sum(X == 1) / float(X.size), 0.2, rtol=1e-4)
    assert np.allclose(np.sum(X == 2) / float(X.size), 0.5, atol=1e-4)


def test_percent_variance():
    X = make_microstructure(n_samples=1, n_phases=3,
                            volume_fraction=(0.3, 0.2, 0.5),
                            percent_variance=.2)
    assert np.allclose(np.sum(X == 1) / float(X.size), 0.09, atol=1e-2)
    assert np.allclose(np.sum(X == 2) / float(X.size), 0.57, atol=1e-2)

def test_3D_volume_fraction():
    X = make_microstructure(n_samples=1, n_phases=3, size=(21, 21, 21),
                           volume_fraction=(0.3, 0.1, 0.6))
    assert np.allclose(np.sum(X == 0) / float(X.size), 0.3, rtol=1e-3)
    assert np.allclose(np.sum(X == 1) / float(X.size), 0.1, rtol=1e-3)

def test_3D_percent_variance():
    X = make_microstructure(n_samples=1, n_phases=3, size=(21, 21, 21),
                            volume_fraction=(0.7, 0.1, 0.2),
                            percent_variance=.2)
    assert np.allclose(np.sum(X == 1) / float(X.size), 0.05, atol=1e-2)
    assert np.allclose(np.sum(X == 2) / float(X.size), 0.2, atol=1e-2)

if __name__ == '__main__':
    test_3D_volume_fraction()
    test_3D_percent_variance()
