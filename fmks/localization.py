from scipy.linalg import lstsq
import numpy as np

from .fext import curry, pipe, array_from_tuple, map

fft = curry(np.fft.fft)

ifft = curry(np.fft.ifft)

fftn = curry(np.fft.rfftn)

ifftn = curry(np.fft.irfftn)

fftshift = curry(np.fft.fftshift)

@curry
def lstsq_slice(fx, fy, redundancy_func, ijk):
    return (ijk + redundancy_func(ijk),
            lstsq(fx[(slice(None),) + ijk + redundancy_func(ijk)],
                  fy[(slice(None),) + ijk],
                  np.finfo(float).eps * 1e4)[0])

def fit_fourier(fx_data, fy_data, redundancy_func):
    lstsq_ijk = lstsq_slice(fx_data, fy_data, redundancy_func)
    return pipe(
        map(lstsq_ijk, np.ndindex(fx_data.shape[1:-1])),
        list,
        array_from_tuple(shape=fx_data.shape[1:], dtype=np.complex)
    )

def faxes(arr):
    return np.arange(arr.ndim - 2) + 1

@curry
def fit_disc(x_data, y_data, redundancy_func):
    return pipe(
        [x_data, y_data],
        map(fftn(axes=faxes(x_data))),
        lambda x: fit_fourier(*x, redundancy_func)
    )

@curry
def fit(x_data, y_data, basis):
    return fit_disc(y_data=y_data, **basis(x_data))

@curry
def predict_disc(x_data, coeff):
    return pipe(
        fftn(x_data, axes=faxes(x_data)),
        lambda x: np.sum(x * coeff[None], axis=-1),
        ifftn(axes=faxes(x_data), s=x_data.shape[1:-1])
    ).real

@curry
def predict(x_data, coeff, basis):
    return predict_disc(basis(x_data)['x_data'], coeff)

def ini_axes(arr):
    return np.arange(arr.ndim - 1)

@curry
def coeff_to_real(coeff, new_shape):
    return pipe(
        coeff,
        ifftn(axes=ini_axes(coeff), s=new_shape),
        fftshift(axes=ini_axes(coeff))
    )
