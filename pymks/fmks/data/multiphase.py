"""Generate a set of random multiphase microstructures using a gaussian blur filter.
"""

import numpy as np
import dask.array as da
from toolz.curried import curry, pipe
from toolz.curried import map as fmap
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
    return pipe(f_data, ifftshift, fftn, lambda x: x * fftn(x_data), ifftn).real


@curry
def _gaussian_blur_filter(grain_size, domain_size):
    return pipe(
        grain_size,
        lambda x: fourier_gaussian(np.ones(x), np.ones(len(x))),
        fftshift,
        zero_pad(shape=domain_size, chunks=None),
    ).compute()


@curry
def quantile_ndim(arr, quantiles):
    """Use np.quantile across varying quantiles

    Args:
      arr: array to quantize (at least 2D)
      quantiles: same dimensions as arr

    Returns:
      quantized array

    >>> a = np.array([np.arange(10), np.arange(10)])
    >>> q = np.array(((0.2, 0.8), (0.41, 0.59)))
    >>> print(quantile_ndim(a, q))
    [[2 7]
     [4 5]]

    """
    return pipe(
        zip(arr, quantiles),
        fmap(lambda x: np.quantile(*x, axis=-1, interpolation="nearest").T),
        np.stack,
    )


@curry
def _segmentation_values(x_data, volume_fraction, percent_variance):
    def func(shape):
        return (2 * np.random.random(shape) - 1) * percent_variance

    def calc_quantiles():
        return pipe(
            np.cumsum(volume_fraction)[:-1], lambda x: x + func((len(x_data), len(x)))
        )

    return pipe(
        x_data,
        lambda x: np.reshape(x, (x_data.shape[0], -1)),
        quantile_ndim(quantiles=calc_quantiles()),
        lambda x: np.reshape(
            x, [x_data.shape[0]] + [1] * (x_data.ndim - 1) + [len(volume_fraction) - 1]
        ),
    )


@curry
def np_generate(grain_size, volume_fraction, percent_variance, x_blur):
    """Construct a microstructure given a random array.

    Args:
      shape (tuple): (n_sample, n_x, n_y, n_z)
      grain_size (tuple): size of the grain size in the microstructure
      volume_fraction (tuple): the percent volume fraction for each phase
      percent_variance (float): the percent variance for each value of
        volume_fraction

    Returns:
      random multiphase microstructures with shape of `shape`

    """

    seg_values = _segmentation_values(
        volume_fraction=volume_fraction, percent_variance=percent_variance
    )

    return sequence(
        _imfilter(f_data=_gaussian_blur_filter(grain_size, x_blur.shape[1:])),
        lambda x: x[..., None] > seg_values(x),
        lambda x: np.sum(x, axis=-1),
    )(x_blur)


@curry
def generate_multiphase(
    shape, grain_size, volume_fraction, chunks=-1, percent_variance=0.0
):
    """Constructs microstructures for an arbitrary number of phases
    given the size of the domain, and relative grain size.

    Args:
      shape (tuple): (n_sample, n_x, n_y, n_z)
      grain_size (tuple): size of the grain size in the microstructure
      volume_fraction (tuple): the percent volume fraction for each phase
      chunks (int): chunks_size of the first
      percent_variance (float): the percent variance for each value of
        volume_fraction

    Returns:
      A dask array of random-multiphase microstructures
      microstructures for the system of shape given by `shape`.

    Example:

    >>> x_tru = np.array([[[0, 0, 0],
    ...                    [0, 1, 0],
    ...                    [1, 1, 1]]])
    >>> da.random.seed(10)
    >>> x = generate_multiphase(
    ...     shape=(1, 3, 3),
    ...     grain_size=(1, 1),
    ...     volume_fraction=(0.5, 0.5)
    ... )
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

    return map_blocks(
        np_generate(grain_size, volume_fraction, percent_variance),
        da.random.random(shape, chunks=(chunks,) + shape[1:]),
        dtype=np.int64,
    )
