import os
import ase
import numpy as np
import ase.io as aio
from toolz.curried import pipe
import pymks.atommks.porosity as pore
from pymks.atommks.grid_generator import generate_grids
from pathlib import Path


def get_pld_lcd(use_fft_method):
    r_Ox = 1.35
    r_Si = 1.35

    len_pixel = 10

    fname = Path(__file__).parent / ".." / "scripts" / "iza_zeolites"/ "MFI.cif"

    rep = [1, 1, 1]
    atom = pipe(fname,
                lambda fname: aio.read(fname),
                lambda x: x.repeat(rep))

    radii={"Si":r_Si, "O": r_Ox}
    grids = generate_grids(
        atom,
        n_pixel=10,
        atomic_radii=radii,
        extend_boundary_atoms=False,
        use_fft_method=use_fft_method
    )

    assert grids['pores'].shape == (202, 198, 133)

    padval = ((1, 1), (1, 1), (0, 0))

    S_dgrid = pipe(grids['pores'],
                   lambda s: np.pad(s, padval, 'constant', constant_values=0),
                   lambda s: pore.dgrid(s, len_pixel=len_pixel))

    return (pore.get_pld(S_dgrid), pore.get_lcd(S_dgrid))


def test_edtGen():
    pld, lcd = get_pld_lcd(False)
    assert np.allclose(pld, 2.61718, atol=1e-3)
    assert np.allclose(lcd, 6.79705, atol=1e-3)


def test_fftGen():
    pld, lcd = get_pld_lcd(True)
    assert np.allclose(pld, 2.6171875, atol=1e-3)
    assert np.allclose(lcd, 6.7230939, atol=1e-3)
