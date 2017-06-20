"""Test the primitive basis.
"""

import numpy as np
from fmks.bases import discretize
from toolz import pipe


def test_local():
    """Local state example.

    An examples where the local states fall between the picks of local
    states. The first specifies the local state space domain between
    `[0, 1]`.
    """
    np.random.seed(4)

    def _compare(x_raw, n_state):
        basis = discretize(n_state=n_state)
        return pipe(
            np.linspace(0, 1, n_state),
            lambda h: np.sum(basis(x_raw) * h[None, None, None, :], axis=-1),
            lambda x: np.allclose(x_raw, x)
        )
    assert _compare(np.random.random((2, 5, 3, 2)), 10)


def test_local_min_max():
    """Local state example with varying min and max.

    An example where the local state space domain is between `[-1,
    1]`.
    """
    basis = discretize(n_state=3, min_=-1)
    assert np.allclose(
        basis(np.array([-1, 0, 1, 0.5])),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]]
    )


def test_error():
    """Test data outside the bounds

    If the local state values in the microstructure are outside of the
    domain they are remapped inside of the domain
    """
    basis = discretize(n_state=2)
    assert np.allclose(
        basis(np.array([-1, 1])),
        [[1, 0], [0, 1]]
    )


def test_many_local_states():
    """Test with many local states
    """
    def _x_data():
        np.random.seed(3)
        return np.random.random((2, 5, 3))

    def _test_data(n_state):
        basis = discretize(n_state=n_state)
        states = lambda: np.linspace(0, 1, n_state)[None, None, None, :]
        return np.sum(basis(_x_data()) * states(), axis=-1)

    assert np.allclose(_x_data(), _test_data(10))
