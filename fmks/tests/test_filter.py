"""Simple filter example
"""

import numpy as np

from toolz import pipe
from fmks.func import dafft, daifft
from fmks.localization import fit, coeff_to_real
from fmks.bases import primitive_basis
import dask.array as da


def _filter(data):
    return np.where(data < 10,
                    np.exp(-abs(data)) * np.cos(data * np.pi),
                    np.exp(-abs(data - 20)) * np.cos((data - 20) * np.pi))


def _coeff(n_space, n_state):
    return np.linspace(1, 0, n_state)[None, :] * \
        _filter(np.linspace(0, 20, n_space))[:, None]


def _fcoeff(n_space, n_state):
    return np.fft.fft(_coeff(n_space, n_state), axis=0)


def _response(x_data, n_space, n_state):
    return pipe(
        np.linspace(0, 1, n_state),
        lambda h: da.maximum(
            1 - abs(x_data[:, :, None] - h) / (h[1] - h[0]),
            0
        ),
        dafft(axis=1),
        lambda fx: da.sum(_fcoeff(n_space, n_state)[None] * fx, axis=-1),
        daifft(axis=1)
    ).real


def _mks_fcoeff(x_data, n_space, n_state):
    return fit(x_data,
               _response(x_data, n_space, n_state),
               primitive_basis(n_state=n_state))


def _mks_coeff(x_data, n_space, n_state):
    return pipe(
        _mks_fcoeff(x_data, n_space, n_state),
        coeff_to_real(new_shape=(n_space,))
    )


def _calc_coeff(n_space, n_state):
    return np.fft.fftshift(_coeff(n_space, n_state), axes=(0,))


def _compare(n_sample, n_space, n_state, chunk):
    np.random.seed(2)
    x_data = da.from_array(np.random.random((n_sample, n_space)),
                           chunks=(chunk, n_space))
    return np.allclose(_calc_coeff(n_space, n_state),
                       _mks_coeff(x_data, n_space, n_state))


def test():
    """Test a simple filter example.
    """
    assert _compare(400, 81, 2, 400)
