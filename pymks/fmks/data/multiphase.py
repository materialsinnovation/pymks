"""Generate a set of random multiphase microstructures using a gaussian blur filter.
"""

import numpy as np
import dask.array as da
from toolz.curried import curry, pipe
from scipy.ndimage.fourier import fourier_gaussian
from pymks.fmks.func import fftn, ifftn, fftshift, ifftshift, zero_pad


conj = curry(np.conjugate)  # pylint: disable=invalid-name
fabs = curry(np.absolute)  # pylint: disable=invalid-name


@curry
def _imfilter(x_data, f_data):
    """
    to convolve f_data over x_data
    """
    return pipe(f_data, ifftshift, fftn, lambda x: conj(x) * fftn(x_data), ifftn, fabs)


@curry
def _gaussian_blur_filter(grain_size, domain_size):
    return pipe(
        grain_size,
        lambda x: fourier_gaussian(np.ones(x), np.ones(len(x))),
        fftshift,
        zero_pad(shape=domain_size, chunks=None),
    ).compute()


@curry
def _cumulative_sum(volume_fraction, n_phases):
    if volume_fraction is None:
        return pipe(n_phases, lambda x: [1.0 / x] * x, lambda x: np.cumsum(x)[:-1])
    if len(volume_fraction) == n_phases:
        if np.sum(volume_fraction) == 1:
            return np.cumsum(volume_fraction)[:-1]
        raise RuntimeError("The terms in the volume fraction list should sum to 1")
    raise RuntimeError("len(volume_fraction) not equal to n_phases.")


@curry
def _segmentation_values(x_data, n_samples, volume_fraction, n_phases):
    return pipe(
        x_data,
        lambda x: np.reshape(x, (n_samples, -1)),
        lambda x: np.quantile(
            x, q=_cumulative_sum(volume_fraction, n_phases), axis=1
        ).T,
        lambda x: np.reshape(x, [n_samples] + [1] * (x_data.ndim - 1) + [n_phases - 1]),
    )


@curry
def _npgenerate(
    n_phases=2, shape=(5, 101, 101), grain_size=(25, 50), volume_fraction=None, seed=10
):
    """
    Generates a microstructure of dimensions of shape and grains
    with dimensions grain_size.

    Returns:
      periodic microstructure
    >>> X_gen = _npgenerate(shape=(1,4,4), grain_size=(4, 4), n_phases=2)
    >>> X_tru = np.array([[[1, 0, 1, 1],
    ...               [0, 0, 0, 1],
    ...               [0, 0, 1, 1],
    ...               [0, 0, 1, 1]]])
    >>> assert np.allclose(X_gen, X_tru)
    """
    np.random.seed(seed)
    seg_values = _segmentation_values(
        n_samples=shape[0], volume_fraction=volume_fraction, n_phases=n_phases
    )

    return pipe(
        shape,
        np.random.random,
        _imfilter(f_data=_gaussian_blur_filter(grain_size, shape[1:])),
        lambda x: x[..., None] >= seg_values(x),
        lambda x: np.sum(x, axis=-1),
    )


# pylint: disable=too-many-arguments
@curry
def generate(
    n_phases=5,
    shape=(5, 101, 101),
    grain_size=(25, 50),
    volume_fraction=None,
    seed=10,
    chunks=(),
):
    """
    Constructs microstructures for an arbitrary number of phases
    given the size of the domain, and relative grain size.
    Returns:
        A dask array of random-multiphase microstructures
        microstructures for the system of shape (n_samples, n_x, ...)
    Example:
    >>> x = generate(shape=(5,11, 11), grain_size=(3,4), n_phases=2, seed=10)
    >>> x.shape
    (5, 11, 11)
    >>> x.chunks
    ((5,), (11,), (11,))
    >>> X_tru = np.array([[[1, 0, 1],
    ...              [1, 1, 0],
    ...              [0, 1, 0]]])
    >>> X_gen = generate(shape=(1,3,3), grain_size=(1, 1), n_phases=2)
    >>> print(X_gen.shape)
    (1, 3, 3)
    >>> print(X_gen.chunks)
    ((1,), (3,), (3,))
    >>> assert np.allclose(X_gen.compute(), X_tru)
    """
    return da.from_array(
        _npgenerate(n_phases, shape, grain_size, volume_fraction, seed),
        chunks=(chunks or (-1,)) + shape[1:],
    )
