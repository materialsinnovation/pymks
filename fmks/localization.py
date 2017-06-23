"""MKS localzation module

The MKS localization module fits data using the Materials Knowledge
System in Fourier Space.

Example:

>>> from fmks.bases import primitive_basis
>>> from fmks.func import allclose

>>> basis = primitive_basis(n_state=2)

>>> x_data = lambda: da.from_array(np.linspace(0, 1, 8).reshape((2, 2, 2)),
...                                chunks=(1, 2, 2))
>>> y_data = lambda: x_data().swapaxes(1, 2)
>>> assert pipe(
...     fit(x_data(), y_data(), basis),
...     predict(x_data(), basis=basis),
...     allclose(y_data())
... )

"""

import dask.array as da
import numpy as np
from scipy.linalg import lstsq
from toolz.curried import pipe
from toolz.curried import map as fmap

from .func import curry, array_from_tuple
from .func import fftshift, rfftn, irfftn
from .func import darfftn


@curry
def lstsq_mks(fx_data, fy_data, redundancy_func, ijk):
    """Do a least square for a single point in the MKS k-space.

    Select a point in k-space using `ijk` tuple and do a least squares
    across samples and local state space to calculate parts of the
    coefficient matrix.

    Args:
      fx_data: microstructure in k-space
      fy_data: response in k-space
      redundancy_func: helps remove redundancies in the coefficient matrix
      ijk: a point in k-space

    Returns:
      a tuple of the coefficient matrix index and vector over local
      state space, which can be used to populate the coefficient
      matrix


    >>> make_data = lambda s: da.from_array(np.arange(np.prod(s),
    ...                                               dtype=float).reshape(s),
    ...                                     chunks=s)

    >>> index, value = lstsq_mks(make_data((2, 1, 1, 3)),
    ...                          make_data((2, 1, 1)),
    ...                          lambda _: (slice(None),),
    ...                          (0, 0))

    >>> print(index)
    (0, 0, slice(None, None, None))
    >>> assert np.allclose(value, [5. / 18., 1. / 9., -1. / 18.])

    """
    fx_data_ = lambda: fx_data[(slice(None),) + ijk + redundancy_func(ijk)]
    fy_data_ = lambda: fy_data[(slice(None),) + ijk]
    return (ijk + redundancy_func(ijk),
            lstsq(fx_data_().compute(),
                  fy_data_().compute(),
                  np.finfo(float).eps * 1e4)[0])


def fit_fourier(fx_data, fy_data, redundancy_func):
    """Fit the data in Fourier space.

    Fit the data after it has been discretized and transformed into
    Fourier space.

    Args:
      fx_data: microstructure in k-space
      fy_data: response in k-space
      redundancy_func: helps remove redundancies in the coefficient matrix

    Returns:
      the coefficient matrix (unchunked)

    >>> make_data = lambda s: da.from_array(np.arange(np.prod(s),
    ...                                               dtype=float).reshape(s),
    ...                                     chunks=s)

    >>> matrix = fit_fourier(make_data((5, 4, 4, 3)),
    ...                      make_data((5, 4, 4)),
    ...                      lambda _: (slice(None),))

    >>> print(matrix.shape)
    (4, 4, 3)

    >>> test_matrix = np.resize([5. / 18., 1. / 9., -1. / 18.], (4, 4, 3))
    >>> assert np.allclose(matrix, test_matrix)

    """
    lstsq_mks_ = lstsq_mks(fx_data, fy_data, redundancy_func)
    return pipe(
        fmap(lstsq_mks_, np.ndindex(fx_data.shape[1:-1])),
        list,
        array_from_tuple(shape=fx_data.shape[1:], dtype=np.complex)
    )


def faxes(arr):
    """Get the spatiol axes to perform the Fourier transform

    The first and the last axes should not have the Fourier transform
    performed.

    Args:
      arr: the discretized array

    Returns:
      an array starting at 1 to n - 2 where n is the length of the
      shape of arr

    >>> faxes(np.array([1]).reshape((1, 1, 1, 1, 1)))
    array([1, 2, 3])

    """
    return np.arange(arr.ndim - 2) + 1


@curry
def fit_disc(x_data, y_data, redundancy_func):
    """Fit the discretized data.

    Fit the data after the data has already been discretized.

    Args:
      x_data: the discretized mircrostructure field
      y_data: the discretized response field
      redundancy_func: helps remove redundancies in the coefficient matrix

    Returns:
      the chunked coefficient matrix based on the chunking of local
      state space from the discretized mircrostructure field


    >>> make_data = lambda s, c: da.from_array(
    ...     np.arange(np.prod(s),
    ...               dtype=float).reshape(s),
    ...     chunks=c
    ... )

    >>> matrix = fit_disc(make_data((6, 4, 4, 3), (2, 4, 4, 1)),
    ...                   make_data((6, 4, 4), (2, 4, 4)),
    ...                   lambda _: (slice(None),))

    >>> print(matrix.shape)
    (4, 3, 3)

    >>> print(matrix.chunks)
    ((4,), (3,), (1, 1, 1))

    >>> assert np.allclose(matrix.compute()[0, 0, 0], 5. / 18.)

    """
    chunks = lambda x: (None,) * (len(x.shape) - 1) + (x_data.chunks[-1],)
    return pipe(
        [x_data, y_data],
        fmap(darfftn(axes=faxes(x_data))),
        list,
        lambda x: fit_fourier(*x, redundancy_func),
        lambda x: da.from_array(x, chunks=chunks(x))
    )


@curry
def fit(x_data, y_data, basis):
    """Calculate the MKS influence coefficients.

    Args:
      x_data: the mircrostructure field
      y_data: the response field
      basis: a function that returns the discretized data and
        redundancy function

    Returns:
      the influence coefficients

    >>> from fmks.bases import primitive_basis

    >>> matrix = fit(da.from_array(np.array([[0], [1]]), chunks=(2, 1)),
    ...              da.from_array(np.array([[2], [1]]), chunks=(2, 1)),
    ...              primitive_basis(n_state=3))
    >>> assert np.allclose(matrix, [[2, 0, 1]])

    """
    return pipe(
        x_data,
        basis,
        lambda x: fit_disc(x[0], y_data, x[1])
    )


@curry
def _predict_disc(x_data, coeff):
    return pipe(
        rfftn(x_data, axes=faxes(x_data)),
        lambda x: np.sum(x * coeff[None], axis=-1),
        irfftn(axes=faxes(x_data), s=x_data.shape[1:-1])
    ).real


@curry
def predict(x_data, coeff, basis):
    """Predict a response given a microstructure

    Args:
      x_data: the microstructure data
      coeff: the influence coefficients
      basis: the basis function

    Returns:
      the response
    """
    return _predict_disc(basis(x_data)[0], coeff)


def _ini_axes(arr):
    return np.arange(arr.ndim - 1)


@curry
def coeff_to_real(coeff, new_shape):
    """Covert the coefficents to real space

    Args:
      coeff: the coefficient in Fourier space
      new_shape: the shape of the coefficients in real space

    Returns:
      the coefficients in real space
    """
    return pipe(
        coeff,
        irfftn(axes=_ini_axes(coeff), s=new_shape),
        fftshift(axes=_ini_axes(coeff))
    )
