import numpy as np
from fmks.fext import ifftn, fftn, curry

def _k_space(size, dx):
    size1 = lambda: (size / 2) if (size % 2 == 0) else (size - 1) / 2
    size2 = lambda: size1() if (size % 2 == 0) else size1() + 1
    k = lambda: np.concatenate(np.arange(size)[:size2()],
                               (np.arange(size) - size1())[:size1()])
    return k() * 2 * np.pi / (dx * size)

def _calc_ksq(x_data, dx):
    i_ = lambda: np.indices(x_data.shape[1:])
    return np.sum(_k_space(x_data.shape[1], dx)[i_()] ** 2, axis=0)[None]

def _axes(x_data):
    return np.arange(len(x_data.shape) - 1) + 1

def _f_response(x_data, dt, gamma, ksq, a1=3., a2=0.):
    FX = lambda: fftn(x_data, _axes(x_data))
    FX3 = lambda: fftn(x_data ** 3, _axes(x_data))
    explicit = lambda: a1 - gamma * a2 * ksq
    implicit = lambda: (1 - gamma * ksq) - explicit
    dt_ = lambda: dt * ksq
    return (FX() * (1 + dt_() * explicit()) - dt_() * FX3()) / (1 - dt_() * implicit())

@curry
def cahn_hilliard_run(x_data, dx, dt, gamma):
    return ifftn(_f_response(_check(x_data),
                             dt,
                             gamma,
                             _calc_ksq(x_data, dx)),
                 _axes(x_data)).real

def _check(x_data):
    if not np.all(np.array(x_data.shape[1:]) == x_data.shape[1]):
        raise RuntimeError("X must represent a square domain")
    return x_data
