"""MKS Correlation Module

For computing auto and cross corelations under assumption
of periodic boundary conditions using discreete fourier
transform.
"""


import dask.array as da
import numpy as np
from toolz.curried import pipe, curry


fft = curry(da.fft.fft)  # pylint: disable=invalid-name

ifft = curry(da.fft.ifft)  # pylint: disable=invalid-name

fftn = curry(da.fft.fftn)  # pylint: disable=invalid-name

ifftn = curry(da.fft.ifftn)  # pylint: disable=invalid-name

fftshift = curry(da.fft.fftshift)  # pylint: disable=invalid-name

ifftshift = curry(da.fft.fftshift)  # pylint: disable=invalid-name

conj = curry(da.conj)  # pylint: disable=invalid-name

func = curry(lambda x, y: conj(x) * fftn(y))  # pylint: disable=invalid-name


def faxes(arr):
    """Get the spatial axes to perform the Fourier transform

    The first axis should not have the Fourier transform
    performed.

    Args:
      arr: the discretized array
    Returns:
      an array starting at 1 to n - 1 where n is the length of the
      shape of arr

    >>> faxes(np.array([1]).reshape((1, 1, 1, 1, 1)))
    (1, 2, 3, 4)
    """
    return tuple(np.arange(arr.ndim - 1) + 1)


@curry
def auto_correlation(arr1):
    """
    Returns auto-corrlation of and input field with itself.
    Args:
        arr1: the input field

    Returns:
        an nd-array of same dimension as the input field

    >>> x_data = np.asarray([[[1, 1, 0], [0, 0, 1], [1, 1, 0]]])
    >>> chunks = x_data.shape
    >>> x_data = da.from_array(x_data, chunks=chunks)
    >>> f_data = auto_correlation(x_data)
    >>> gg = [[[0.33333333, 0.22222222, 0.33333333],
    ...        [0.22222222, 0.55555556, 0.22222222],
    ...        [0.33333333, 0.22222222, 0.33333333]]]
    >>> assert np.allclose(f_data.compute(), gg)
    """
    return pipe(arr1,
                fftn(axes=faxes(arr1)),
                lambda x: conj(x) * x,
                ifftn(axes=faxes(arr1)),
                fftshift(axes=faxes(arr1)),
                lambda x: x.real / arr1[0].size)


@curry
def cross_correlation(arr1, arr2):
    """
    Returns the cross-correlation of and input field with another field.
    Args:
        arr1: the input field
        arr2: the other input field

    Returns:
        an nd-array of same dimension as the input field

    >>> x_data = np.asarray([[[1,1,0], [0,0,1], [1,1,0]]])
    >>> chunks = x_data.shape
    >>> x_data = da.from_array(x_data, chunks=chunks)
    >>> y_data = da.from_array(1 - x_data, chunks=chunks)
    >>> f_data = cross_correlation(x_data, y_data)
    >>> gg = np.asarray([[[ 2.22222222e-01,  3.33333333e-01,  2.22222222e-01],
    ...                   [ 3.33333333e-01, -2.63163976e-16,  3.33333333e-01],
    ...                   [ 2.22222222e-01,  3.33333333e-01,  2.22222222e-01]]])
    >>> assert np.allclose(f_data.compute(), gg)
    """
    return pipe(arr1,
                fftn(axes=faxes(arr1)),
                lambda x: conj(x) * fftn(arr2, axes=faxes(arr2)),
                ifftn(axes=faxes(arr1)),
                fftshift(axes=faxes(arr1)),
                lambda x: x.real / arr1[0].size)
