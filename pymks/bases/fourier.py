import numpy as np
from .abstract import _AbstractMicrostructureBasis


class FourierBasis(_AbstractMicrostructureBasis):

    r"""
    Discretize a continuous field into `deg` local states using complex
    exponentials such that,

    .. math::

       \frac{1}{\Delta} \int_s m(h, x) dx =
       \sum_{- L / 2}^{L / 2} m[l, s] exp(l*h*I)

    and the local state space :math:`H` is mapped into the orthogonal domain
    .. math::

       0 \le  H \le 2 \pi

    The mapping of :math:`H` into the domain is done automatically in PyMKS by
    using the `domain` key work argument.

    >>> n_states = 3
    >>> X = np.array([[0., np.pi / 3],
    ...               [2 * np.pi / 3, 2 * np.pi]])
    >>> X_result = np.array(([[[1, 1, 1],
    ...                        [1, np.exp(np.pi / 3 * 1j),
    ...                         np.exp(- np.pi / 3 * 1j)]],
    ...                       [[1, np.exp(2 *np.pi / 3 * 1j),
    ...                         np.exp(- 2 * np.pi / 3 * 1j)],
    ...                         [1, 1, 1]]]))
    >>> basis = FourierBasis(n_states, [0., 2 * np.pi])
    >>> assert(np.allclose(basis.discretize(X), X_result))

    If the microstructure local state values fall outside of the period of
    the specified domain, the values will be mapped back into the domain.

    >>> n_states = 2
    >>> X = np.array([[0, 1.5]])
    >>> four_basis = FourierBasis(n_states, domain=[0, 1])
    >>> X_result = np.array([[[1, 1],
    ...                       [1, -1]]])
    >>> assert np.allclose(X_result, four_basis.discretize(X))
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

        >>> X = np.array([[-1, -1./2, 0.,  1./2, 1]])
        >>> f_basis = FourierBasis(4, [-1, 1.])
        >>> X_result = np.array([[[1, 1, 1, 1],
        ...                       [1, 1j, -1j, -1],
        ...                       [1, -1, -1, 1],
        ...                       [1, -1j, 1j, -1],
        ...                       [1, 1, 1, 1]]])
        >>> assert(np.allclose(X_result, f_basis.discretize(X)))

        """
        X_scaled = 2. * np.pi * ((X.astype(float) - self.domain[0]) /
                                 (self.domain[1] - self.domain[0]))
        nones = ([None for i in X.shape])
        X_states = np.zeros(X_scaled.shape + (self.n_states,))
        X_states[..., :] = ((np.arange(self.n_states + 1) / 2)[1:] *
                            (-1) ** np.arange(1, self.n_states + 1))[nones]
        return np.exp(X_scaled[..., None] * X_states * 1j)
