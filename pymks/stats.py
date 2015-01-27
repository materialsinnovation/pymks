import numpy as np
from .filter import Correlation

"""
The SpatialStatisticsModel takes in a microstructure and returns its two
point statistics. Current the funciton only work for interger valued
microstructures and the DiscreteIndicatorBasis.
"""


def autocorrelate(X_, periodic_axes=[]):
    """
    Computes the autocorrelation from a microstructure function.

    Test non-periodic autocorrelation.

    >>> n_states = 2
    >>> X = np.array([[[0, 0, 0],
    ...                [0, 1, 0],
    ...                [0, 0, 0]]])
    >>> from pymks.bases import DiscreteIndicatorBasis
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> X_auto = autocorrelate(X_, periodic_axes=(0, 1))
    >>> X_test = np.array([[[0., 0., 0.],
    ...                   [0., 1./9, 0.],
    ...                   [0., 0., 0.]]])
    >>> assert(np.allclose(np.real_if_close(X_auto[0, ..., 1]), X_test[0]))

    Args:
      X_: microstructure funciton
      periodic_axes: axes that are periodic. (0, 2) would indicate
          that axes x and z are periodic in a 3D microstrucure.
    Returns:
      Autocorrelations for microstructure function X_.
    """
    s = Fkernel_shape(X_, periodic_axes)
    corr = Correlation(X_, Fkernel_shape=s).convolve(X_)
    return truncate(corr, X_.shape[:-1]) / normalize(X_, s)


def crosscorrelate(X_, periodic_axes=[]):
    """
    Computes the crosscorrelations from a microstructure function.

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
      X_: microstructure funciton
      periodic_axes: axes that are periodic. (0, 2) would indicate
          that axes x and z are periodic in a 3D microstrucure.
    Returns:
      Crosscorelations for microstructure function X_.
    """

    n_states = X_.shape[-1]
    Niter = n_states // 2
    Nslice = n_states * (n_states - 1) / 2
    s = Fkernel_shape(X_, periodic_axes)
    tmp = [Correlation(X_,
                       Fkernel_shape=s).convolve(np.roll(X_, i,
                                                         axis=-1)) for i
           in range(1, Niter + 1)]
    corr = np.concatenate(tmp, axis=-1)[..., :Nslice]
    return truncate(corr, X_.shape[:-1]) / normalize(X_, s)


def correlate(X_, periodic_axes=[]):
    """
    Computes the autocorrelations and crosscorrelations from a microstructure
    function.

    Args:
      X_: microstructure funciton
      periodic_axes: axes that are periodic. (0, 2) would indicate
          that axes x and z are periodic in a 3D microstrucure.
    Returns:
      Autocorrelations and crosscorrelations for microstructure funciton X_.
    """
    X_auto = autocorrelate(X_, periodic_axes=periodic_axes)
    X_cross = crosscorrelate(X_, periodic_axes=periodic_axes)
    return np.concatenate((X_auto, X_cross), axis=-1)


def normalize(X_, Fkernel_shape):
    """
    Returns the normalization for the statistics

    The normalization should be Nx * Ny in the center of the domain.

    >>> Nx = Ny = 5
    >>> X_ = np.zeros((1, Nx, Ny, 1))
    >>> Fkernel_shape = np.array((2 * Nx, Ny))
    >>> norm =  normalize(X_, Fkernel_shape)
    >>> assert norm.shape == (1, Nx, Ny, 1)
    >>> assert np.allclose(norm[0, Nx / 2, Ny / 2, 0], 25)

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
        return truncate(corr.convolve(ones), X_.shape[:-1])


def Fkernel_shape(X_, periodic_axes):
    """
    Returns the shape of the kernel in Fourier space with non-periodic padding.

    >>> Nx = Ny = 5
    >>> X_ = np.zeros((1, Nx, Ny, 1))
    >>> periodic_axes = [1]
    >>> assert (Fkernel_shape(X_, periodic_axes=periodic_axes) == [8, 5]).all()

    Args:
      X_ : microstructure funciton
      periodic_axes: the axes of the array that are periodic

    Returns:
      Shape of the new Fkernel array

    """
    axes = np.arange(len(X_.shape) - 2) + 1
    a = np.ones(len(axes), dtype=float) * 1.75
    a[list(periodic_axes)] = 1
    return (np.array(X_.shape)[axes] * a).astype(int)


def truncate(a, shape):
    """
    Truncates the edges of the array, a, based on the shape. This is
    used to unpad a padded convolution.

    >>> print truncate(np.arange(10).reshape(1, 10, 1), (1, 5))[0, ..., 0]
    [3 4 5 6 7]
    >>> print truncate(np.arange(9).reshape(1, 9, 1), (1, 5))[0, ..., 0]
    [2 3 4 5 6]
    >>> print truncate(np.arange(10).reshape((1, 10, 1)), (1, 4))[0, ..., 0]
    [3 4 5 6]
    >>> print truncate(np.arange(9).reshape((1, 9, 1)), (1, 4))[0, ..., 0]
    [2 3 4 5]

    >>> a = np.arange(5 * 4).reshape((1, 5, 4, 1))
    >>> print truncate(a, shape=(1, 3, 2))[0, ..., 0]
    [[ 5  6]
     [ 9 10]
     [13 14]]

    >>> a = np.arange(5 * 4 * 3).reshape((1, 5, 4, 3, 1))
    >>> assert (truncate(a, (1, 2, 2, 1))[0, ..., 0]  ==
    ...         [[[16], [19]], [[28], [31]]]).all()

    """
    a_shape = np.array(a.shape)
    n = len(shape)
    new_shape = a_shape.copy()
    new_shape[:n] = shape
    diff_shape = a_shape - new_shape
    index0 = (diff_shape + (diff_shape % 2) * (new_shape % 2)) / 2
    index1 = index0 + new_shape
    multi_slice = tuple(slice(index0[ii], index1[ii]) for ii in range(n))
    return a[multi_slice]
