import numpy as np
from .abstract import _AbstractMicrostructureBasis


class LegendreBasis(_AbstractMicrostructureBasis):
    """
    Discretize a continuous field into `deg` local states using a
    Legendre polynomial basis.

    >>> deg = 3
    >>> X = np.array([[0.25, 0.1],
    ...               [0.5, 0.25]])
    >>> X_Legendre = np.array([[[-0.3125, -0.75, 0.5],
    ...                         [ 1.15,   -1.2, 0.5]],
    ...                        [[-1.25,      0, 0.5],
    ...                         [-0.3125, -0.75, 0.5]]])
    >>> basis = LegendreBasis(3, [0., 0.5])
    >>> assert(np.allclose(basis.discretize(X), X_Legendre))

    """

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
          X: field representing the microstructure
        Returns:
          field of local states
        """
        self.check(X)
        leg = np.polynomial.legendre
        X_scaled = 2. * X - self.domain[0] - self.domain[1] / (self.domain[1] -
                                                               self.domain[0])
        norm = (2. * np.arange(self.n_states) + 1) / 2.
        X_Legendre = np.flipud(leg.legval(X_scaled,
                                          np.eye(self.n_states) * norm))
        return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))
