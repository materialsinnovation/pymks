import numpy as np
from .abstract import _AbstractMicrostructureBasis


class ContinuousIndicatorBasis(_AbstractMicrostructureBasis):
    """
    Discretize a continuous field into `n_states` local states. The
    field must be between 0 and 1. For example, if a cell has a value
    of 0.4 and `n_states=2` then the local state is (0.6, 0.4) (the
    local state must sum to 1).

    >>> n_states = 10
    >>> np.random.seed(4)
    >>> X = np.random.random((2, 5, 3, 2))
    >>> X_ = ContinuousIndicatorBasis(n_states=n_states).discretize(X)
    >>> H = np.linspace(0, 1, n_states)
    >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
    >>> assert np.allclose(X, Xtest)


    """
    def __init__(self, n_states=10):
        """
        Instantiate a `ContinuousIndicatorBasis`

        Args:
          n_states: The number of local states

        """
        self.n_states = n_states

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
          X: continuous field between 0 and 1
        Returns:
          field of local states between 0 and 1
        """
        self.check(X, [0, 1])
        H = np.linspace(0, 1, self.n_states)
        return np.maximum(1 - (abs(X[..., None] - H)) / (H[1] - H[0]), 0)
