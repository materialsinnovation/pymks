from .filter import Correlation
import numpy as np

"""
The stats functions take in a microstructure function and returns its two
point statistics.
"""


def autocorrelate(X, basis, periodic_axes=[], n_jobs=1, confidence_index=None,
                  autocorrelations=None):
    """
    Computes the autocorrelation from a microstructure function.

    Args:
        X (ND array): The microstructure, an `(n_samples, n_x, ...)`
            shaped array where `n_samples` is the number of samples and
            `n_x` is the spatial discretization.
        basis (class): an instance of a bases class
        periodic_axes (list, optional): axes that are periodic. (0, 2) would
            indicate that axes x and z are periodic in a 3D microstrucure.
        n_jobs (int, optional): number of parallel jobs to run. only used if
            pyfftw is install.
        confidence_index (ND array, optional): array with same shape as X used
            to assign a confidence value for each data point.
        autocorrelations (list, optional): list of spatial autocorrelations to
            be computed corresponding to the states in basis.n_states. For
            example, if basis.n_states=[0, 2], then autocorrelations=[(0, 0),
            (2, 2)] computes the autocorrelations for the states 0 and 2. If
            no list is passed, all autocorrelations in basis.n_states are
            computed.

    Returns:
        Autocorrelations for a microstructure.

    Non-periodic example

    >>> n_states = 2
    >>> X = np.array([[[0, 0, 0],
    ...                [0, 1, 0],
    ...                [0, 0, 0]]])
    >>> from pymks.bases import PrimitiveBasis
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> X_auto = autocorrelate(X, prim_basis, periodic_axes=(0, 1))
    >>> X_test = np.array([[[0., 0., 0.],
    ...                   [0., 1./9, 0.],
    ...                   [0., 0., 0.]]])
    >>> assert(np.allclose(np.real_if_close(X_auto[0, ..., 1]), X_test[0]))
    """
    if periodic_axes is None:
        periodic_axes = []
    if autocorrelations is None:
        autocorrelations = _auto_correlations(basis.n_states)
    else:
        autocorrelations = _correlations_to_indices(autocorrelations, basis)
    return _compute_stats(X, basis, autocorrelations, confidence_index,
                          periodic_axes, n_jobs)


def crosscorrelate(X, basis, periodic_axes=None, n_jobs=1,
                   confidence_index=None, crosscorrelations=None):
    """
    Computes the crosscorrelations from a microstructure function.

    Args:
        X (ND array): The microstructure, an `(n_samples, n_x, ...)`
            shaped array where `n_samples` is the number of samples and
            `n_x` is the spatial discretization.s.
        basis (class): an instance of a bases class
        periodic_axes (list, optional): axes that are periodic. (0, 2) would
            indicate that axes x and z are periodic in a 3D microstrucure.
        n_jobs (int, optional): number of parallel jobs to run. only used if
            pyfftw is install.
        confidence_index (ND array, optional): array with same shape as X used
            to assign a confidence value for each data point.
        crosscorrelations (list, optional): list of cross-correlations to
            be computed corresponding to the states in basis.n_states. For
            example if basis.n_states=[2, 4, 6] then crosscorrelations=[(2, 4),
            (2, 6)] computes the cross-correlations with local states 2 and 4
            as well as 2 and 6. If no list is passed, all cross-correlations
            in basis.n_states are computed.

    Returns:
        Crosscorelations for a microstructure.

    Examples

    Test for 2 states.

    >>> n_states = 2
    >>> X = np.array([[[0, 1, 0],
    ...                [0, 1, 0],
    ...                [0, 1, 0]]])
    >>> from pymks.bases import PrimitiveBasis
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> X_cross = crosscorrelate(X, prim_basis, periodic_axes=[0, 1])
    >>> X_test = np.array([[[[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]],
    ...                     [[1/3.], [0.], [1/3.]]]])
    >>> assert(np.allclose(X_cross, X_test))

    Test for 3 states

    >>> n_states = 3
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> assert(crosscorrelate(X, prim_basis,
    ...        periodic_axes=[0, 1]).shape == (1, 3, 3, 3))

    Test for 4 states

    >>> n_states = 4
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> assert(crosscorrelate(X, prim_basis,
    ...        periodic_axes=[0, 1]).shape == (1, 3, 3, 6))

    Test for 5 states

    >>> n_states = 5
    >>> prim_basis = PrimitiveBasis(n_states=n_states)
    >>> assert(crosscorrelate(X, prim_basis,
    ...        periodic_axes=[0, 1]).shape == (1, 3, 3, 10))
    """
    if periodic_axes is None:
        periodic_axes = []
    if crosscorrelations is None:
        crosscorrelations = _cross_correlations(basis.n_states)
    else:
        crosscorrelations = _correlations_to_indices(crosscorrelations,
                                                     basis)
    return _compute_stats(X, basis, crosscorrelations, confidence_index,
                          periodic_axes, n_jobs)


def correlate(X, basis, periodic_axes=None, n_jobs=1,
              confidence_index=None, correlations=None):
    """
    Computes the autocorrelations and crosscorrelations from a microstructure
    function.

    Args:
        X (ND array): The microstructure, an `(n_samples, n_x, ...)`
            shaped array where `n_samples` is the number of samples and
            `n_x` is the spatial discretization.
        basis (class): an instance of a bases class
        periodic_axes (list, optional): axes that are periodic. (0, 2) would
            indicate that axes x and z are periodic in a 3D microstrucure.
        n_jobs (int, optional): number of parallel jobs to run. only used if
            pyfftw is install.
        confidence_index (ND array, optional): array with same shape as X used
            to assign a confidence value for each data point.
        correlations (list, optional): list of  spatial _check_shapes to
            be computed corresponding to the states in basis.n_states. For
            example, it n_states=[0, 2, 5] [(0, 0), (2, 2), (0, 5)] computes
            the autocorrelations with local states 0 and 2 as well as the
            cross-correlation between 0 and 5. If no list is passed, all
            spatial correlations are computed.

    Returns:
        Autocorrelations and crosscorrelations for a microstructure.

    Example

    >>> from pymks import PrimitiveBasis
    >>> prim_basis = PrimitiveBasis(2, [0, 1])
    >>>
    >>> np.random.seed(0)
    >>> X = np.random.randint(2, size=(1, 3))
    >>> X_corr = correlate(X, prim_basis)
    >>> X_result = np.array([[0, 0.5, 0],
    ...                      [1 / 3., 2 / 3., 0],
    ...                      [0, 0.5, 0.5]])
    >>> assert np.allclose(X_corr, X_result)
    """
    if periodic_axes is None:
        periodic_axes = []
    if correlations is None:
        L = basis.n_states
        _auto, _cross = _auto_correlations(L), _cross_correlations(L)
        correlations = (_auto[0] + _cross[0], _auto[1] + _cross[1])
    else:
        correlations = _correlations_to_indices(correlations, basis)
    return _compute_stats(X, basis, correlations, confidence_index,
                          periodic_axes, n_jobs)


def _compute_stats(X, basis, correlations, confidence_index,
                   periodic_axes, n_jobs):
    """Helper function to compute statistics

    Args:
        X (ND array): The microstructure, an `(n_samples, n_x, ...)`
            shaped array where `n_samples` is the number of samples and
            `n_x` is the spatial discretization.
        basis: an instance of a bases class
        correlations: list of  spatial correlations to be computed.
        confidence_index: array with same shape as X used to assign a
            confidence value for each data point.
        periodic_axes: axes that are periodic. (0, 2) would indicate that axes
            and z are periodic in a 3D microstrucure.
        n_jobs (int, optional): number of parallel jobs to run.
    """
    if max(max(correlations)) > len(basis.n_states) + 1:
        raise ValueError(('values in correlations are larger than') +
                         ('the length of basis.n_states'))
    X_ = basis.discretize(X)
    X_ = _mask_X_(X_, confidence_index)
    basis._n_jobs = n_jobs
    _Fkernel_shape(X.shape, basis, periodic_axes)
    _norm = _normalize(X_.shape, basis, confidence_index)
    return _correlate(X_, basis, correlations) / _norm


def _correlate(X_, basis, correlations):
    """
    Helper function used to calculate the unnormalized correlation counts.

    Args:
        X_ (ND array): The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        basis (class): an instance of a bases class.
        correlations (list): list of correlations to compute. `[(0, 0),
            (1, 1)]`

    Returns:
        correlation counts for a given microstructure function

    Example

    >>> from pymks.datasets import make_microstructure
    >>> from pymks.bases import PrimitiveBasis
    >>> X = make_microstructure(n_samples=2, n_phases=3,
    ...                         size=(2, 2), grain_size=(2, 2), seed=99)
    >>> prim_basis = PrimitiveBasis(n_states=3, domain=[0, 2])
    >>> X_ = prim_basis.discretize(X)
    >>> correlations = (tuple(prim_basis.n_states), tuple(prim_basis.n_states))
    >>> X_corr = _correlate(X_, prim_basis, correlations=correlations)
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
    corr = Correlation(X_[..., correlations[0]],
                       basis).convolve(X_[..., correlations[1]])
    return _truncate(corr, X_.shape[:-1])


def _auto_correlations(n_states):
    """Returns list of autocorrelations

    Args:
        n_states: number of local states

    Returns:
        list of tuples for autocorrelations

    >>> l = _auto_correlations(np.arange(3))
    >>> assert l == ((0, 1, 2), (0, 1, 2))
    """
    return tuple(n_states), tuple(n_states)


def _cross_correlations(n_states):
    """Returns list of crosscorrelations

    Args:
        n_states: number of local states

    Returns:
        list of tuples for crosscorrelations

    >>> l = _cross_correlations(np.arange(3))
    >>> assert l == ((0, 0, 1), (1, 2, 2))
    """
    l = range(len(n_states))
    cross_corr = [[(l[i], l[j]) for j in l[1:][i:]] for i in l[:-1]]
    flat_corr = [item for sublist in cross_corr for item in sublist]
    l_0 = tuple([_l[0] for _l in flat_corr])
    l_1 = tuple([_l[1] for _l in flat_corr])
    return l_0, l_1


def _normalize(X_shape, basis, confidence_index):
    """
    Returns the normalization for the statistics

    The normalization should be Nx * Ny in the center of the domain.

    Args:
        `X_`: The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.
        basis (class): an instance of a bases class
        confidence_index (ND array, optional): array with same shape as X used
            to assign a confidence value for each data point.

    Returns:
        Normalization

    """

    if basis._axes_shape == X_shape[1:] and confidence_index is None:
        return float(np.prod(X_shape[1:]))
    else:
        mask = confidence_index
        if mask is None:
            mask = np.ones(X_shape[1:-1])[None]
        corr = Correlation(mask[..., None], basis)
        return _truncate(corr.convolve(mask[..., None]), X_shape)


def _Fkernel_shape(X_shape, basis, periodic_axes):
    """
    Assigns the shape of the kernel in Fourier space with non-periodic padding
    to the basis.

    Args:
        `X_shape`: The shape of discretized microstructure function,
            `(n_samples, n_x, ..., n_states)` where `n_samples` is the number
            of samples, `n_x` is the spatial discretization, and n_states is
            the number of local states.
        basis: an instance of a bases class
        periodic_axes: the axes of the array that are periodic

    Example

    >>> Nx = Ny = 5
    >>> X_ = np.zeros((1, Nx, Ny, 1))
    >>> periodic_axes = [1]
    >>> from pymks import PrimitiveBasis
    >>> p_basis = PrimitiveBasis(2)
    >>> p_basis._axes = np.array([1, 2])
    >>> _Fkernel_shape(X_.shape, p_basis, periodic_axes=periodic_axes)
    >>> assert p_basis._axes_shape == (10, 5)
    """
    a = np.ones(len(basis._axes), dtype=float) * 2
    a[list(periodic_axes)] = 1
    basis._axes_shape = tuple((np.array(X_shape)[basis._axes] * a).astype(int))


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

    >>> print(_truncate(np.arange(10).reshape(1, 10, 1), (1, 5))[0, ..., 0])
    [3 4 5 6 7]
    >>> print(_truncate(np.arange(9).reshape(1, 9, 1), (1, 5))[0, ..., 0])
    [2 3 4 5 6]
    >>> print(_truncate(np.arange(10).reshape((1, 10, 1)), (1, 4))[0, ..., 0])
    [3 4 5 6]
    >>> print(_truncate(np.arange(9).reshape((1, 9, 1)), (1, 4))[0, ..., 0])
    [2 3 4 5]

    >>> a = np.arange(5 * 4).reshape((1, 5, 4, 1))
    >>> print(_truncate(a, shape=(1, 3, 2))[0, ..., 0])
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


def _correlations_to_indices(correlations, basis):
    """
    Helper function to select correct indices given the local state values in
    basis.n_states.

    Args:
        correlations: list of correlations to be computed
        basis: an instance of a basis class.

    Returns:
        list of correlations in terms of indices
    """
    try:
        l_0 = tuple([list(basis.n_states).index(_l[0]) for _l in correlations])
        l_1 = tuple([list(basis.n_states).index(_l[1]) for _l in correlations])
    except ValueError as ve:
        raise ValueError('correlations value ' + ve.message[0] +
                         ' is not in basis.n_states')
    return (l_0, l_1)
