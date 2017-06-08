import numpy as np

from fmks.fext import pipe, fft, ifft
from fmks.localization import fit, coeff_to_real
from fmks.bases import primitive_basis

def filter(x):
    return np.where(x < 10,
                    np.exp(-abs(x)) * np.cos(x * np.pi),
                    np.exp(-abs(x - 20)) * np.cos((x - 20) * np.pi))

def coeff(n_space, n_state):
    return np.linspace(1, 0, n_state)[None,:] * \
        filter(np.linspace(0, 20, n_space))[:,None]

def fcoeff(n_space, n_state):
    return np.fft.fft(coeff(n_space, n_state), axis=0)

def response(x_data, n_space, n_state):
    return pipe(
        np.linspace(0, 1, n_state),
        lambda h: np.maximum(1 - abs(x_data[:,:,None] - h) / (h[1] - h[0]), 0),
        fft(axis=1),
        lambda fx: np.sum(fcoeff(n_space, n_state)[None] * fx, axis=-1),
        ifft(axis=1)
    ).real

def mks_fcoeff(x_data, n_space, n_state):
    return fit(x_data,
               response(x_data, n_space, n_state),
               primitive_basis(n_state=n_state))

def mks_coeff(x_data, n_space, n_state):
    return pipe(
        mks_fcoeff(x_data, n_space, n_state),
        coeff_to_real(new_shape=(n_space,))
    )

def calc_coeff(n_space, n_state):
    return np.fft.fftshift(coeff(n_space, n_state), axes=(0,))

def compare(n_sample, n_space, n_state):
    np.random.seed(2)
    x_data = np.random.random((n_sample, n_space))
    return np.allclose(calc_coeff(n_space, n_state),
                       mks_coeff(x_data, n_space, n_state))

def test():
    assert compare(400, 81, 2)
