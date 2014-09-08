import numpy as np
from .abstract import _AbstractMicrostructureBasis


class LegendreBasis(_AbstractMicrostructureBasis):
    r"""
    Discretize a continuous field into `deg` local states using a
    Legendre polynomial basis such that,

    .. math::

       m_h \left[i\right] = \frac{2h +1}{2} P_h \left(
       2 \left( \frac{\phi\left[i\right] - \chi_{\text{min}}}{\chi_{\text{max}}
        - \chi_{\text{min}}} \right) - 1
       \right)

    where the :math:`P_h` are Legendre polynomials and

    .. math::

       \chi_{\text{min}} \le \phi \le \chi_{\text{max}}

    and :math:`0 \le h \le n-1`.

    >>> n_states = 3
    >>> X = np.array([[0.25, 0.1],
    ...               [0.5, 0.25]])
    >>> def P(x):
    ...    x = 4 * x - 1
    ...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
    ...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
    ...    return np.rollaxis(tmp, 0, 3)[:,:,::-1]
    >>> basis = LegendreBasis(n_states, [0., 0.5])
    >>> assert(np.allclose(basis.discretize(X), P(X)))

    """

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
          X: field representing the microstructure
        Returns:
          field of local states

        >>> X = np.array([[-1, 1],
        ...               [0, -1]])
        >>> basis = LegendreBasis(3, [-1, 1])
        >>> def P(x):
        ...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
        ...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
        ...    return np.rollaxis(tmp, 0, 3)[:,:,::-1]
        >>> assert(np.allclose(basis.discretize(X), P(X)))

        """
        self.check(X)
        leg = np.polynomial.legendre
        X_scaled = (2. * X - self.domain[0] - self.domain[1]) /\
                   (self.domain[1] - self.domain[0])
        norm = (2. * np.arange(self.n_states) + 1) / 2.
        X_Legendre = (leg.legval(X_scaled, np.eye(self.n_states) * norm))[::-1]
        return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))
