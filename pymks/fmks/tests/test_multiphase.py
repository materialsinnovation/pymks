"""Test the multiphase microstructure generator

"""

import numpy as np

import pytest
import dask.array as da

from pymks.fmks.data.multiphase import generate


def test_chunking():
    """Test that the generated microstructue is chunked correctly

    """
    da.random.seed(10)
    data = generate(
        shape=(5, 11, 11), grain_size=(3, 4), volume_fraction=(0.5, 0.5), chunks=2
    )
    assert data.shape == (5, 11, 11)
    assert data.chunks == ((2, 2, 1), (11,), (11,))


def test_2d():
    """Regression test for microstructure phases
    """
    da.random.seed(10)
    data = generate(shape=(1, 4, 4), grain_size=(4, 4), volume_fraction=(0.5, 0.5))
    assert np.allclose(data, [[[0, 0, 0, 0], [1, 0, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0]]])


def test_1d():
    """Test that 1D works
    """
    da.random.seed(10)
    data = generate(shape=(1, 10), grain_size=(4,), volume_fraction=(0.5, 0.5))
    assert np.allclose(data, [0, 0, 0, 0, 1, 0, 1, 1, 1, 1])


def test_grain_size():
    """
    Test incompatible grain_size and shape
    """
    with pytest.raises(RuntimeError) as excinfo:
        da.random.seed(10)
        generate(shape=(1, 10, 10), grain_size=(4,), volume_fraction=(0.5, 0.5))
    assert str(excinfo.value) == "`shape` should be of length `len(grain_size) + 1`"


def test_volume_fraction():
    """Test incoherent volume_fraction
    """
    with pytest.raises(RuntimeError) as excinfo:
        da.random.seed(10)
        generate(shape=(1, 10), grain_size=(2,), volume_fraction=(0.4, 0.4, 0.4))
    assert str(excinfo.value) == "The terms in the volume fraction list should sum to 1"


def test_3d():
    """Test 3D

    Also tests for bug that occured when chunks and sample size are
    different.

    """
    da.random.seed(10)
    data = generate(
        shape=(5, 5, 5, 5),
        grain_size=(1, 5, 1),
        volume_fraction=(0.3, 0.3, 0.4),
        chunks=2,
    )
    assert data.chunks == ((2, 2, 1), (5,), (5,), (5,))
    assert np.allclose(
        data[0, 1],
        [
            [0, 1, 0, 2, 2],
            [0, 1, 0, 2, 2],
            [0, 1, 1, 2, 1],
            [0, 1, 2, 2, 0],
            [0, 1, 2, 2, 0],
        ],
    )
