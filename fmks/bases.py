import numpy as np
from .fext import curry


def _discretize(x_data, states):
    return 1 - (abs(x_data[..., None] - states)) / (states[1] - states[0])

@curry
def primitive(x_data, n_states, min_=0., max_=1.):
    return np.maximum(
        _discretize(x_data,
                   np.linspace(min_, max_, n_states)),
        0)
