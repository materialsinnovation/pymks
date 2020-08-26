"""Test functions for linear elastic FE
"""

import pytest
import numpy as np
from toolz.curried import pipe, first, get
from pymks.fmks.data.elastic_fe import solve_fe


def test_3d():
    """Test FE in 3D
    """

    def setone(arr):
        arr[0, :, (arr.shape[0] - 1) // 2] = 1.0
        return arr

    assert pipe(
        5,
        lambda x: np.zeros((1, x, x, x), dtype=int),
        setone,
        solve_fe(elastic_modulus=(1.0, 10.0), poissons_ratio=(0.0, 0.0)),
        lambda x: np.allclose(
            [np.mean(x["strain"][0, ..., i]) for i in range(6)],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    )


def test_3d_bcs():
    """Test FE in 3D different macro strain
    """
    np.random.seed(8)

    def testing(size, macro_strain):
        return pipe(
            size,
            lambda x: np.random.randint(2, size=(1, x, x, x)),
            solve_fe(
                elastic_modulus=(10.0, 1.0),
                poissons_ratio=(0.3, 0.3),
                macro_strain=macro_strain,
            ),
            get("displacement"),
            first,
            lambda x: (
                np.allclose(x[-1, ..., 0] - x[0, ..., 0], size * macro_strain),
                np.allclose(x[0, ..., 1], x[-1, ..., 1]),
            ),
            all,
        )

    assert testing(4, 0.1)


def test_issue106():
    """Test for missing phases
    """

    def test(x_data):
        solve_fe(x_data, elastic_modulus=(1, 2, 3), poissons_ratio=(0.3, 0.3, 0.3))

    size = 5
    test(np.zeros((1, size, size, size), dtype=int))
    test(np.ones((1, size, size, size), dtype=int))

    with pytest.raises(RuntimeError) as excinfo:
        x_data = np.ones((1, size, size, size), dtype=int)
        x_data[0, 0, 0] = -1
        test(x_data)

    assert str(excinfo.value) == "X must be between 0 and 2."
