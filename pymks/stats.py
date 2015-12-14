import numpy as np
from .filter import Correlation

"""
The stats functions take in a microstructure function and returns its two
point statistics.
"""


def autocorrelate(X_, periodic_axes=[], confidence_index=None,
                  autocorrelations=None):
    """
    Computes the autocorrelation from a microstructure function.

    Args:
        X_ (ND array): The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        periodic_axes (list, optional): axes that are periodic. (0, 2) would
            indicate that axes x and z are periodic in a 3D microstrucure.
        confidence_index (ND array, optional): array with same shape as X used
            to assign a confidence value for each data point.
        autocorrelations (list, optional): list of spatial autocorrelatiions to
            be computed. For example [(0, 0), (1, 1)] computes the
            autocorrelations with local states 0 and 1. If no list is passed,
            all autocorrelations are computed.

    Returns:
        Autocorrelations for microstructure function `X_`.

    Non-periodic example

    >>> n_states = 2
    >>> X = np.array([[[0, 0, 0],
    ...                [0, 1, 0],
    ...                [0, 0, 0]]])
    >>> from pymks.bases import PrimitiveBasis
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> X_ = prim_basis.discretize(X)
    >>> X_auto = autocorrelate(X_, periodic_axes=(0, 1))
    >>> X_test = np.array([[[0., 0., 0.],
    ...                   [0., 1./9, 0.],
    ...                   [0., 0., 0.]]])
    >>> assert(np.allclose(np.real_if_close(X_auto[0, ..., 1]), X_test[0]))
    """
    if periodic_axes is None:
        periodic_axes = []
    if autocorrelations is None:
        correlations = _auto_correlations(X_.shape[-1])
    X_ = _mask_X_(X_, confidence_index)
    s = _Fkernel_shape(X_, periodic_axes)
    auto = _correlate(X_, s, correlations)
    return auto / _normalize(X_, s, confidence_index)


def _correlate(X_, s, correlations):
    """
    Helper function used to calculate the unnormalized correlation counts.

    Args:
        X_ (ND array): The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        s (tuple): shape of the Fkernel used for the convolution

    Returns:
        correlation counts for a given microstructure function

    Example

    >>> from pymks.datasets import make_microstructure
    >>> from pymks.bases import PrimitiveBasis
    >>> X = make_microstructure(n_samples=2, n_phases=3,
    ...                         size=(2, 2), grain_size=(2, 2), seed=99)
    >>> prim_basis = PrimitiveBasis(n_states=3, domain=[0, 2])
    >>> X_ = prim_basis.discretize(X)
    >>> correlations = [(l, l) for l in range(3)]
    >>> X_corr = _correlate(X_, X_.shape[1:-1], correlations=correlations)
    >>> X_result = np.array([[[[0, 0, 0],
    ...                        [0, 0, 2]],
    ...                       [[0, 0, 0],
    ...                        [1, 1, 2]]],
    ...                      [[[0, 0, 0],
    ...                        [0, 0, 0]],
    ...                       [[2, 0, 2],
    ...                        [2, 0, 2]]]])
    >>> assert np.allclose(X_result, X_corr)
    """

    l_0, l_1 = [l[0] for l in correlations], [l[1] for l in correlations]
    corr = Correlation(X_[..., l_0], Fkernel_shape=s).convolve(X_[..., l_1])
    return _truncate(corr, X_.shape[:-1])


def crosscorrelate(X_, periodic_axes=None, confidence_index=None,
                   crosscorrelations=None):
    """
    Computes the crosscorrelations from a microstructure function.

    Args:
        X_ (ND array): The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        periodic_axes (list, optional): axes that are periodic. (0, 2) would
            indicate that axes x and z are periodic in a 3D microstrucure.
        confidence_index (ND array, optional): array with same shape as X used
            to assign a confidence value for each data point.
        crosscorrelations (list, optional): list of cross-correlatiions to
            be computed. For example [(0, 1), (0, 2)] computes the
            cross-correlations with local states 0 and 1 as well as 0 and 2.
            If no list is passed, all cross-correlations are computed.

    Returns:
        Crosscorelations for microstructure function `X_`.

    Examples

    Test for 2 states.

    >>> n_states = 2
    >>> X = np.array([[[0, 1, 0],
    ...                [0, 1, 0],
    ...                [0, 1, 0]]])
    >>> from pymks.bases import PrimitiveBasis
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> X_ = prim_basis.discretize(X)
    >>> X_cross = crosscorrelate(X_, periodic_axes=[0, 1])
    >>> X_test = np.array([[[[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]]]])
    >>> assert(np.allclose(X_cross, X_test))

    Test for 3 states

    >>> n_states = 3
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> X_ = prim_basis.discretize(X)
    >>> assert(crosscorrelate(X_, periodic_axes=[0, 1]).shape == (1, 3, 3, 3))

    Test for 4 states

    >>> n_states = 4
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> X_ = prim_basis.discretize(X)
    >>> assert(crosscorrelate(X_, periodic_axes=[0, 1]).shape == (1, 3, 3, 6))

    Test for 5 states

    >>> n_states = 5
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> X_ = prim_basis.discretize(X)
    >>> assert(crosscorrelate(X_, periodic_axes=[0, 1]).shape == (1, 3, 3, 10))
    """
    if periodic_axes is None:
        periodic_axes = []
    if crosscorrelations is None:
        correlations = _cross_correlations(X_.shape[-1])
    X_ = _mask_X_(X_, confidence_index)
    s = _Fkernel_shape(X_, periodic_axes)
    cross = _correlate(X_, s, correlations)
    return cross / _normalize(X_, s, confidence_index)


def correlate(X_, periodic_axes=None,
              confidence_index=None, correlations=None):
    """
    Computes the autocorrelations and crosscorrelations from a microstructure
    function.

    Args:
        X_ (ND array): The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        periodic_axes (list, optional): axes that are periodic. (0, 2) would
            indicate that axes x and z are periodic in a 3D microstrucure.
        confidence_index (ND array, optional): array with same shape as X used
            to assign a confidence value for each data point.
        correlations (list, optional): list of  spatial correlatiions to
            be computed. For example [(0, 0), (1, 1), (0, 2)] computes the
            autocorrelations with local states 0 and 1 as well as the
            cross-correlation between 0 and 2. If no list is passed, all
            spatial correlations are computed.

    Returns:
        Autocorrelations and crosscorrelations for microstructure funciton
        `X_`.

    Example

    >>> from pymks import PrimitiveBasis
    >>> prim_basis = PrimitiveBasis(2, [0, 1])
    >>>
    >>> np.random.seed(0)
    >>> X = np.random.randint(2, size=(1, 3))
    >>> X_ = prim_basis.discretize(X)
    >>> X_corr = correlate(X_)
    >>> X_result = np.array([[0, 0.5, 0],
    ...                      [1 / 3., 2 / 3., 0],
    ...                      [0, 0.5, 0.5]])
    >>> assert np.allclose(X_corr, X_result)
    """
    if periodic_axes is None:
        periodic_axes = []
    if correlations is None:
        L = X_.shape[-1]
        correlations = _auto_correlations(L) + _cross_correlations(L)
    X_ = _mask_X_(X_, confidence_index)
    s = _Fkernel_shape(X_, periodic_axes)
    corr = _correlate(X_, s, correlations)
    return corr / _normalize(X_, s, confidence_index)


def _auto_correlations(n_states):
    """Returns list of autocorrelations

    Args:
        n_states: number of local states

    Returns:
        list of tuples for autocorrelations

    >>> l = _auto_correlations(3)
    >>> assert l == [(0, 0), (1, 1), (2, 2)]
    """
    local_states = range(n_states)
    return [(l, l) for l in local_states]


def _cross_correlations(n_states):
    """Returns list of crosscorrelations

    Args:
        n_states: number of local states

    Returns:
        list of tuples for crosscorrelations

    >>> l = _cross_correlations(3)
    >>> assert l == [(0, 1), (0, 2), (1, 2)]
    """
    l = range(n_states)
    cross_corr = [[(l[i], l[j]) for j in l[1:][i:]] for i in l[:-1]]
    return [item for sublist in cross_corr for item in sublist]


def _normalize(X_, s, confidence_index):
    """
    Returns the normalization for the statistics

    The normalization should be Nx * Ny in the center of the domain.

    Args:
        `X_`: The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        _Fkernel_shape : the shape of the kernel is Fourier space (array)
        confidence_index: array with same shape as X used to assign a
            confidence value for each data point.

    Returns:
        Normalization

    Example

    >>> Nx = Ny = 5
    >>> X_ = np.zeros((1, Nx, Ny, 1))
    >>> _Fkernel_shape  = np.array((2 * Nx, Ny))
    >>> norm =  _normalize(X_, _Fkernel_shape , None)
    >>> assert norm.shape == (1, Nx, Ny, 1)
    >>> assert np.allclose(norm[0, Nx / 2, Ny / 2, 0], 25)
    """

    if (s == X_.shape[1:-1]).all() and confidence_index is None:
        return float(np.prod(X_.shape[1:-1]))
    else:
        mask = confidence_index
        if mask is None:
            mask = np.ones(X_.shape[1:-1])[None]
        corr = Correlation(mask[..., None], Fkernel_shape=s)
        return _truncate(corr.convolve(mask[..., None]), X_.shape[:-1])


def _Fkernel_shape(X_, periodic_axes):
    """
    Returns the shape of the kernel in Fourier space with non-periodic padding.

    Args:
        `X_`: The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        periodic_axes: the axes of the array that are periodic

    Returns:
        shape of the new Fkernel array

    Example

    >>> Nx = Ny = 5
    >>> X_ = np.zeros((1, Nx, Ny, 1))
    >>> periodic_axes = [1]
    >>> assert (_Fkernel_shape(X_,
    ...                        periodic_axes=periodic_axes) == [8, 5]).all()
    """
    axes = np.arange(len(X_.shape) - 2) + 1
    a = np.ones(len(axes), dtype=float) * 1.75
    a[list(periodic_axes)] = 1
    return (np.array(X_.shape)[axes] * a).astype(int)


def _truncate(a, shape):
    """
    _truncates the edges of the array, a, based on the shape. This is
    used to unpad a padded convolution.

    Args:
        a: array to be truncated
        shape: new shape of array

    Returns:
        truncated array

    Example

    >>> print _truncate(np.arange(10).reshape(1, 10, 1), (1, 5))[0, ..., 0]
    [3 4 5 6 7]
    >>> print _truncate(np.arange(9).reshape(1, 9, 1), (1, 5))[0, ..., 0]
    [2 3 4 5 6]
    >>> print _truncate(np.arange(10).reshape((1, 10, 1)), (1, 4))[0, ..., 0]
    [3 4 5 6]
    >>> print _truncate(np.arange(9).reshape((1, 9, 1)), (1, 4))[0, ..., 0]
    [2 3 4 5]

    >>> a = np.arange(5 * 4).reshape((1, 5, 4, 1))
    >>> print _truncate(a, shape=(1, 3, 2))[0, ..., 0]
    [[ 5  6]
     [ 9 10]
     [13 14]]

    >>> a = np.arange(5 * 4 * 3).reshape((1, 5, 4, 3, 1))
    >>> assert (_truncate(a, (1, 2, 2, 1))[0, ..., 0]  ==
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


def _mask_X_(X_, confidence_index):
    """
    Helper function to verify that the confidence_index is the correct
    shape.

    Args:
        `X_`: The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        confidence_index: array with same shape as X used to assign a
            confidence value for each data point.

    Returns:
        The discretized microstructure function scaled by the confidence_index.

    """
    if confidence_index is not None:
        if X_.shape[:-1] != confidence_index.shape:
            raise RuntimeError('confidence_index does not match shape of X')
        X_ = X_ * confidence_index[..., None]
    return X_
