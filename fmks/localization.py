"""MKS localzation module

The MKS localization module fits data using the Materials Knowledge
System in Fourier Space.

Example:

>>> from fmks.bases import primitive_basis
>>> from fmks.fext import allclose

>>> basis = primitive_basis(n_state=2)

>>> x_data = lambda: np.linspace(0, 1, 4).reshape((1, 2, 2))
>>> y_data = lambda: x_data().swapaxes(1, 2)
>>> assert pipe(
...     fit(x_data(), y_data(), basis),
...     predict(x_data(), basis=basis),
...     allclose(y_data())
... )

"""


from scipy.linalg import lstsq
import numpy as np

from .fext import curry, pipe, array_from_tuple, fmap
from .fext import fftshift, rfftn, irfftn


@curry
def _lstsq_slice(fx_data, fy_data, redundancy_func, ijk):
    return (ijk + redundancy_func(ijk),
            lstsq(fx_data[(slice(None),) + ijk + redundancy_func(ijk)],
                  fy_data[(slice(None),) + ijk],
                  np.finfo(float).eps * 1e4)[0])


def _fit_fourier(fx_data, fy_data, redundancy_func):
    lstsq_ijk = _lstsq_slice(fx_data, fy_data, redundancy_func)
    return pipe(
        fmap(lstsq_ijk, np.ndindex(fx_data.shape[1:-1])),
        list,
        array_from_tuple(shape=fx_data.shape[1:], dtype=np.complex)
    )


def _faxes(arr):
    return np.arange(arr.ndim - 2) + 1


@curry
def _fit_disc(y_data, x_data, redundancy_func):
    return pipe(
        [x_data, y_data],
        fmap(rfftn(axes=_faxes(x_data))),
        lambda x: _fit_fourier(*x, redundancy_func)
    )


@curry
def fit(x_data, y_data, basis):
    """Calculate the MKS influence coefficients.

    Args:
      x_data: the mircrostructure data
      y_data: the response field
      basis: a function that returns the discretized data and
        redundancy function

    Returns:
      the influence coefficients
    """
    return _fit_disc(y_data, *basis(x_data))


@curry
def _predict_disc(x_data, coeff):
    return pipe(
        rfftn(x_data, axes=_faxes(x_data)),
        lambda x: np.sum(x * coeff[None], axis=-1),
        irfftn(axes=_faxes(x_data), s=x_data.shape[1:-1])
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
