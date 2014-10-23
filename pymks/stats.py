import numpy as np
from .filter import Correlation

"""
The SpatialStatisticsModel takes in a microstructure and returns its two
point statistics. Current the funciton only work for interger valued
microstructures and the DiscreteIndicatorBasis.
"""


def autocorrelate(X_, periodic_axes=[]):
    """
    Computes the autocorrelation for a microstructure
    """
    s = Fkernel_shape(X_, periodic_axes)
    return Correlation(X_).convolve(X_) / normalize(X_, s)


def crosscorrelate(X_, periodic_axes=[]):
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
    >>> X_cross = crosscorrelate(X_, periodic_axes=[0, 1])
    >>> X_test = np.array([[[[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]]]])
    >>> assert(np.allclose(X_cross, X_test))

    Test for 3 states

    >>> n_states = 3
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> assert(crosscorrelate(X_, periodic_axes=[0, 1]).shape == (1, 3, 3, 3))

    Test for 4 states

    >>> n_states = 4
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> assert(crosscorrelate(X_, periodic_axes=[0, 1]).shape == (1, 3, 3, 6))

    Test for 5 states

    >>> n_states = 5
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> assert(crosscorrelate(X_, periodic_axes=[0, 1]).shape == (1, 3, 3, 10))

    Args:
      X: microstructure
    Returns:
      Crosscorelations for microstructure X.
    '''
    
    n_states = X_.shape[-1]
    Niter = n_states // 2
    Nslice = n_states * (n_states - 1) / 2

    s = Fkernel_shape(X_, periodic_axes)
    tmp = [Correlation(X_, Fkernel_shape=s).convolve(np.roll(X_, i,
                                                             axis=-1)) for i in range(1,
                                                                                      Niter + 1)]
    return np.concatenate(tmp, axis=-1)[..., :Nslice] / normalize(X_, s)

def normalize(X_, Fkernel_shape):
    """
    Returns the normalization for the statistics

    The normalization should be Nx * Ny in the center of the domain.
    
    >>> Nx = Ny = 5
    >>> X_ = np.zeros((1, Nx, Ny, 1))
    >>> Fkernel_shape = np.array((2 * Nx, Ny))
    >>> norm =  normalize(X_, Fkernel_shape)
    >>> assert norm.shape == (1, 2 * Nx, Ny, 1)
    >>> assert np.allclose(norm[0, Nx, Ny / 2, 0], 25)
    
    Args:
      X_: discretized microstructure (array)
      Fkernel_shape: the shape of the kernel is Fourier space (array)
    Returns:
      Normalization 
      
    """
    if (Fkernel_shape == X_.shape[1:-1]).all():
        return np.prod(X_.shape[1:-1])
    else:
        ones = np.ones(X_.shape)
        corr = Correlation(ones, Fkernel_shape=Fkernel_shape)
        return corr.convolve(ones)

def Fkernel_shape(X_, periodic_axes):
    """
    Returns the shape of the kernel in Fourier space with non-periodic padding.

    >>> Nx = Ny = 5
    >>> X_ = np.zeros((1, Nx, Ny, 1))
    >>> periodic_axes = [1]
    >>> print Fkernel_shape(X_, periodic_axes=periodic_axes)
    [8 5]
    
    """
    axes = np.arange(len(X_.shape) - 2) + 1
    a = np.ones(len(axes), dtype=float) * 1.6
    a[periodic_axes] = 1
    return (np.array(X_.shape)[axes] * a).astype(int)


