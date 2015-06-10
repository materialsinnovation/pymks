import numpy as np
from .indicator import _Indicator


class ContinuousIndicatorBasis(_Indicator):

    r"""
    Discretize a continuous field into `n_states` local states such
    that:

    .. math::

       \frac{1}{\Delta x \Delta t} \int_{H} \int_{s} \int_{n} \Lambda(h - l)
       m(h, x, t) dx dt dh = m[l, s, n]

    where :math:`\Lambda` is the primitive basis (also called hat function)

    .. math::

       \Lambda (h - l) = max \Bigg (1-\Bigg |\frac{h(L - 1)}{H} -
       \frac{Hl}{L-1} \Bigg|, 0\Bigg)


    The field must be between the specified domain values. For
    example, if a cell has a value of 0.4, n_states=2 and the
    domain=[0, 1] then the local state is (0.6, 0.4) (the local state
    must sum to 1).

    Example

    >>> n_states = 10
    >>> np.random.seed(4)
    >>> X = np.random.random((2, 5, 3, 2))
    >>> X_ = ContinuousIndicatorBasis(n_states, [0, 1]).discretize(X)
    >>> H = np.linspace(0, 1, n_states)
    >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
    >>> assert np.allclose(X, Xtest)

    Check that the basis works for values outside of 0 and 1.

    >>> n_states = 3
    >>> X = np.array([-1, 0, 1, 0.5])
    >>> X_test = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]]
    >>> X_ = ContinuousIndicatorBasis(n_states, [-1, 1]).discretize(X)
    >>> assert np.allclose(X_, X_test)

    """

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is thes patial discretization.
        Returns:
            Float valued field of local states between 0 and 1.
        """
        self.check(X)
        H = np.linspace(self.domain[0], self.domain[1], self.n_states)
        return np.maximum(1 - (abs(X[..., None] - H)) / (H[1] - H[0]), 0)
