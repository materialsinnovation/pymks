"""Test the primitive basis.
"""

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


    # >>> n_states = 10
    # >>> np.random.seed(4)
    # >>> X = np.random.random((2, 5, 3, 2))
    # >>> X_ = PrimitiveBasis(n_states, [0, 1]).transform(X)
    # >>> H = np.linspace(0, 1, n_states)
    # >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
    # >>> assert np.allclose(X, Xtest)

    # Here is an example where the local state space domain is between `[-1, 1]`.

    # >>> n_states = 3
    # >>> X = np.array([-1, 0, 1, 0.5])
    # >>> X_test = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]]
    # >>> X_ = PrimitiveBasis(n_states, [-1, 1]).transform(X)
    # >>> assert np.allclose(X_, X_test)

    # If the local state values in the microstructure are outside of the domain
    # they can no longer be represented by two primitive basis functions and
    # violates constraint above.

    # >>> n_states = 2
    # >>> X = np.array([-1, 1])
    # >>> prim_basis = PrimitiveBasis(n_states, domain=[0, 1])
    # >>> prim_basis.transform(X)
    # Traceback (most recent call last):
    # ...
    # RuntimeError: x_raw must be within the specified domain
