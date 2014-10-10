import numpy as np
from .filter import Correlation

"""
The SpatialStatisticsModel takes in a microstructure and returns its two
point statistics. Current the funciton only work for interger valued
microstructures and the DiscreteIndicatorBasis.
"""


def autocorrelate(X_):
    """
    Computes the autocorrelation for a microstructure
    """
    return Correlation(X_).convolve(X_)


def crosscorrelate(X_):
    '''
    Computes the crosscorrelations for a microstructure.

    Test for 2 states.

    >>> n_states = 2
    >>> X = np.array([[[0, 1, 0],
    ...                [0, 1, 0],
    ...                [0, 1, 0]]])
    >>> from pymks.bases import DiscreteIndicatorBasis
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> X_cross = crosscorrelate(X_)
    >>> X_test = np.array([[[[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]]]])
    >>> assert(np.allclose(X_cross, X_test))

    Test for 3 states

    >>> n_states = 3
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> assert(crosscorrelate(X_).shape == (1, 3, 3, 3))

    Test for 4 states

    >>> n_states = 4
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> assert(crosscorrelate(X_).shape == (1, 3, 3, 6))

    Test for 5 states

    >>> n_states = 5
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> assert(crosscorrelate(X_).shape == (1, 3, 3, 10))

    Args:
      X: microstructure
    Returns:
      Crosscorelations for microstructure X.
    '''

    n_states = X_.shape[-1]
    Niter = n_states // 2
    Nslice = n_states * (n_states - 1) / 2
    tmp = [Correlation(X_).convolve(np.roll(X_, i,
                                             axis=-1)) for i in range(1,
                                                                      Niter + 1)]
    return np.concatenate(tmp, axis=-1)[..., :Nslice]
