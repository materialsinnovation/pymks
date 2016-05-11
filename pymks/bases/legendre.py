from .real_ffts import _RealFFTBasis
import numpy as np


class LegendreBasis(_RealFFTBasis):

    r"""
    Discretize a continuous field into `deg` local states using a
    Legendre polynomial basis such that,

    .. math::

       \frac{1}{\Delta x} \int_s m(h, x) dx =
       \sum_0^{L-1} m[l, s] P_l(h)

    where the :math:`P_l` are Legendre polynomials and the local state space
    :math:`H` is mapped into the orthogonal domain of the Legendre polynomials

    .. math::

       -1 \le  H \le 1

    The mapping of :math:`H` into the domain is done automatically in PyMKS by
    using the `domain` key work argument.

    >>> n_states = 3
    >>> X = np.array([[0.25, 0.1],
    ...               [0.5, 0.25]])
    >>> def P(x):
    ...    x = 4 * x - 1
    ...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
    ...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
    ...    return np.rollaxis(tmp, 0, 3)
    >>> basis = LegendreBasis(n_states, [0., 0.5])
    >>> assert(np.allclose(basis.discretize(X), P(X)))

    If the microstructure local state values fall outside of the specified
    domain they will no longer be mapped into the orthogonal domain of the
    legendre polynomials.

    >>> n_states = 2
    >>> X = np.array([-1, 1])
    >>> leg_basis = LegendreBasis(n_states, domain=[0, 1])
    >>> leg_basis.discretize(X)
    Traceback (most recent call last):
    ...
    RuntimeError: X must be within the specified domain

    """

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
        Returns:
            Float valued field of of Legendre polynomial coefficients.

        >>> X = np.array([[-1, 1],
        ...               [0, -1]])
        >>> leg_basis = LegendreBasis(3, [-1, 1])
        >>> def p(x):
        ...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
        ...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
        ...    return np.rollaxis(tmp, 0, 3)
        >>> assert(np.allclose(leg_basis.discretize(X), p(X)))

        """
        self.check(X)
        self._select_axes(X)
        leg = np.polynomial.legendre
        X_scaled = (2. * X - self.domain[0] - self.domain[1]) /\
                   (self.domain[1] - self.domain[0])
        norm = (2. * np.array(self.n_states) + 1) / 2.
        X_Legendre = (leg.legval(X_scaled, np.eye(len(self.n_states)) * norm))
        return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))
