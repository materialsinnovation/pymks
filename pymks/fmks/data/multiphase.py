"""Generate a set of random multiphase microstructures using a gaussian blur filter.
"""

import numpy as np
import dask.array as da
from toolz.curried import curry, pipe
from scipy.ndimage.fourier import fourier_gaussian
from ..func import (
    fftn,
    ifftn,
    fftshift,
    ifftshift,
    zero_pad,
    sequence,
    map_blocks,
)


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
def _segmentation_values(x_data, volume_fraction):
    return pipe(
        x_data,
        lambda x: np.reshape(x, (x_data.shape[0], -1)),
        lambda x: np.quantile(x, q=np.cumsum(volume_fraction)[:-1], axis=1).T,
        lambda x: np.reshape(
            x, [x_data.shape[0]] + [1] * (x_data.ndim - 1) + [len(volume_fraction) - 1]
        ),
    )


@curry
def generate(shape, grain_size, volume_fraction, chunks=-1):
    """Constructs microstructures for an arbitrary number of phases
    given the size of the domain, and relative grain size.

    Args:
      shape (tuple): (n_sample, n_x, n_y, n_z)
      grain_size (tuple): size of the grain size in the microstructure
      volume_fraction (tuple): the percent volume fraction for each phase
      chunks (int): chunks_size of the first

    Returns:
      A dask array of random-multiphase microstructures
      microstructures for the system of shape (n_samples, n_x, ...)

    Example:

    >>> x_tru = np.array([[[0, 1, 0],
    ...                    [0, 1, 0],
    ...                    [1, 1, 1]]])
    >>> da.random.seed(10)
    >>> x = generate(shape=(1, 3, 3), grain_size=(1, 1), volume_fraction=(0.5, 0.5))
    >>> print(x.shape)
    (1, 3, 3)
    >>> print(x.chunks)
    ((1,), (3,), (3,))
    >>> assert np.allclose(x, x_tru)

    """

    if len(grain_size) + 1 != len(shape):
        raise RuntimeError("`shape` should be of length `len(grain_size) + 1`")

    if not np.allclose(np.sum(volume_fraction), 1):
        raise RuntimeError("The terms in the volume fraction list should sum to 1")

    seg_values = _segmentation_values(volume_fraction=volume_fraction)

    np_generate = sequence(
        _imfilter(f_data=_gaussian_blur_filter(grain_size, shape[1:])),
        lambda x: x[..., None] >= seg_values(x),
        lambda x: np.sum(x, axis=-1),
    )

    return map_blocks(
        np_generate,
        da.random.random(shape, chunks=(chunks,) + shape[1:]),
        dtype=np.int64,
    )
