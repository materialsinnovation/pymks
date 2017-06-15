"""Test the localization model.
"""

import numpy as np
from fmks.bases import primitive_basis
from fmks.localization import fit


def _get_x():
    return np.linspace(0, 1, 4).reshape((1, 2, 2))


def test():
    """Very simple example.
    """
    assert np.allclose(fit(_get_x(),
                           _get_x().swapaxes(1, 2),
                           primitive_basis(n_state=2)),
                       [[[0.5, 0.5],
                         [-2, 0]],
                        [[-0.5, 0],
                         [-1, 0]]])
