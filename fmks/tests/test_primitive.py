"""Test the primitive basis.
"""

import pytest
import numpy as np
from fmks.bases import discretize
from fmks.fext import pipe


def test_local():
    """Local state example.

    An examples where the local states fall between the picks of local
    states. The first specifies the local state space domain between
    `[0, 1]`.
    """
    np.random.seed(4)

    def _compare(x_raw, n_state):
        # pylint: disable= no-value-for-parameter
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
    basis = discretize(n_state=3, min_=-1)  # pylint: disable= no-value-for-parameter
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
