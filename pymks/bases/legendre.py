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

    def __init__(self, deg, domain):
        """
        Instantiate a Legendre polynomial basis.
        
        Args:
            deg: Degree of the polynomial basis.
            domain: If a basis is specified, the domain must be used
               indicate the range of expected values for the microstructure.
        """
        self.deg = deg
        self.domain = domain
        
    def discretize(self, X):
        """
        Discretize `X`.
        
        Args:
          X: field representing the microstructure
        Returns:
          field of local states
        """
        self.check(X, self.domain)
        leg = np.polynomial.legendre
        X_scaled = 2. * X - self.domain[0] - self.domain[1] / (self.domain[1] - self.domain[0])
        norm = (2. * np.arange(self.deg) + 1) / 2.
        X_Legendre = np.flipud(leg.legval(X_scaled, np.eye(self.deg) * norm))
        return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))

