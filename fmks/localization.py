from scipy.linalg import lstsq
import numpy as np

from .fext import curry, pipe, array_from_tuple, map

fftn = curry(np.fft.rfftn)

ifftn = curry(np.fft.irfftn)

def s1(ijk):
    if np.all(np.array(ijk) == 0):
        return (slice(None),)
    else:
        return (slice(-1),)

def s0():
    return (slice(None),)

def rcond():
    return np.finfo(float).eps * 1e4

@curry
def lstsq_slice(fx, fy, ijk):
    return (ijk + s1(ijk),
            lstsq(fx[s0() + ijk + s1(ijk)],
                  fy[s0() + ijk],
                  rcond())[0])

def ffit(fx_data, fy_data):
    lstsq_ijk = lstsq_slice(fx_data, fy_data)
    return pipe(
        map(lstsq_ijk, np.ndindex(fx_data.shape[1:-1])),
        list,
        array_from_tuple(shape=fx_data.shape[1:], dtype=np.complex)
    )

def faxes(x_data):
    return np.arange(x_data.ndim - 2) + 1

@curry
def fit(x_data, y_data):
    return pipe(
        [x_data, y_data],
        map(fftn(axes=faxes(x_data))),
        lambda x: ffit(*x)
    )

@curry
def predict(x_data, coeff):
    return pipe(
        x_data,
        fftn(axes=faxes(x_data)),
        lambda x: np.sum(x * coeff[None], axis=-1),
        ifftn(axes=faxes(x_data), s=x_data.shape[1:-1])
    ).real

def coeff_to_real(coeff, new_shape):
    axes = np.arange(coeff.ndim - 1)
    rcoeff = ifftn(coeff.copy(), axes=axes, s=new_shape)
    return np.fft.fftshift(rcoeff, axes=axes)
