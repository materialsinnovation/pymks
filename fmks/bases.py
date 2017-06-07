import numpy as np
from .fext import curry


def _discretize(x_data, states):
    return 1 - (abs(x_data[..., None] - states)) / (states[1] - states[0])

def discretize(x_data, n_state, min_=0., max_=1.):
    return np.maximum(
        _discretize(x_data,
                    np.linspace(min_, max_, n_state)),
        0)

def redundancy(ijk):
     if np.all(np.array(ijk) == 0):
         return (slice(None),)
     else:
         return (slice(-1),)

@curry
def primitive_basis(x_data, n_state, min_=0., max_=1.):
    return {'x_data' : discretize(x_data, n_state, min_=min_, max_=max_),
            'redundancy_func' : redundancy}
