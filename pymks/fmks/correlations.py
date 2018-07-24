"""MKS Correlation Module

For computing auto and cross corelations under assumption
of periodic boundary conditions using discreete fourier
transform.
"""


import numpy as np
from toolz.curried import pipe, curry
from .func import dafftshift, dafftn, daifftn, daconj


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


def corr_master(arr1, arr2):
    """
    Returns cross correlation between the two input fields, arr1 and arr2
    """
    return pipe(arr1,
                dafftn(axes=faxes(arr1)),
                lambda x: daconj(x) * dafftn(arr2, axes=faxes(arr2)),
                daifftn(axes=faxes(arr1)),
                dafftshift(axes=faxes(arr1)),
                lambda x: x.real)


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
    return corr_master(arr1, arr1) / arr1[0].size


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
    return corr_master(arr1, arr2) / arr1[0].size
