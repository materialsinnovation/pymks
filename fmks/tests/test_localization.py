from fmks.bases import primitive_basis
import numpy as np
from fmks.localization import fit


def get_x():
    return np.linspace(0, 1, 4).reshape((1, 2, 2))


def test():
    assert np.allclose(fit(get_x(),
                           get_x().swapaxes(1, 2),
                           primitive_basis(n_state=2)),
                       [[[0.5,  0.5],
                         [-2,   0]],
                        [[-0.5, 0],
                         [-1,   0]]])
