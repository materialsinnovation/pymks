"""
@author: ashanker9@gatech.edu
"""
import ase
import numpy as np
from .helpers import *
from toolz.curried import curry, pipe, merge


@curry
def get_scaled_positions(coords, cell, pbc, wrap=True):
    """Get positions relative to unit cell.
    If wrap is True, atoms outside the unit cell will be wrapped into
    the cell in those directions with periodic boundary conditions
    so that the scaled coordinates are between zero and one."""

    fractional = np.linalg.solve(cell.T, coords.T).T

    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional = np.mod(fractional, 1.0)
    return fractional


@curry
def get_real_positions(coords, cell):
    """Return real space coordinates
    given fractional coordiantes and
    cell parameters"""
    return np.dot(cell.T, coords.T).T


@curry
def AtomCenters(coords, box, len_pixel):
    """
    Args: Coordinates of all atoms [ndArray], dimension of box, pixel size

    returns: Atom Center location voxels
    """

    atom_centers = np.zeros(coords.shape)
    for dim in range(3):
        atom_centers[:,dim] = pipe(coords[:,dim],
                                   lambda x: np.around(x*len_pixel))
    return atom_centers


def generate_grids(ase_atom, atomic_radii=None, n_pixel=10, extend_boundary_atoms=False, use_fft_method=False):
    """Generate voxalized representation corresponding to pore volume and
    the atomic species.

    Args:
      atom: an ASE atom object
      atomic_radii
      n_pixel: number of pixels per unit length
      extend_boundary_atoms: whether to have boundary atoms cutoff at
        the boundary
      use_fft_method: whether to use the FFT method of the EDT method
        for generating the atom volumes

    Returns:
      A dictionary with keys of "pore" and atom type and values of
      binary numpy arrays. The arrays represent whether the atom type
      or pore exists in each voxel. Each voxel can only occupy one of the
      states returned.

    """

    atom = ase_atom
    len_pixel = n_pixel

    dgnls = atom.cell.diagonal()
    if np.any(dgnls == 0):
        coords = atom.get_positions()
        cmax = coords.max(axis=0)
        cmin = coords.min(axis=0)
        dgnls = cmax - cmin

    coords = pipe(atom,
                  lambda x: x.get_positions(),
                  lambda x: np.mod(x, dgnls),
                  lambda x: x - x.min(axis=0))

    box_dim  = np.ceil((coords.max(axis=0)) * len_pixel).astype(int) + 1
    sym_list = np.array(atom.get_chemical_symbols())
    syms = np.unique(sym_list)

    atom_centers = AtomCenters(coords, box_dim, len_pixel)

    max_r = atomic_radii[max(atomic_radii, key=atomic_radii.get)] # max atom radius

    scaler = np.asarray([len_pixel * (2 * max_r+1)] * 3)

    spheres = {}
    if use_fft_method:
        for symbol in atomic_radii:
            spheres[symbol] = sphere(atomic_radii[symbol] * len_pixel)
        generator = generator_fft(box_dim=box_dim, len_pixel=len_pixel, full=extend_boundary_atoms, scaler=scaler)
    else:
        spheres = atomic_radii.copy()
        generator = generator_edt(box_dim=box_dim, len_pixel=len_pixel, full=extend_boundary_atoms, scaler=scaler)

    S = None
    S_list = []
    for sym in syms:
        c = atom_centers[sym_list == sym]
        atom_r = spheres[sym]
        indxs = [c[:, dim].astype(int) for dim in range(3)]
        S_list.append(generator(indxs, atom_r))
        if S is None:
            S = S_list[0].copy()
        else:
            S += S_list[-1]

    S = (S < 1e-2) * 1

    return merge(dict(pores=S), dict(zip(syms, S_list)))

@curry
def generator_edt(indxs, atom_r, box_dim, len_pixel, full=False, scaler=0.0):

    S= np.ones(box_dim)
    S[indxs[0], indxs[1], indxs[2]] = 0

    if full:
        scaled_box_dim = (box_dim + scaler)
        S = padder(S, scaled_box_dim, 1)

    S = transform_edt(S.astype(np.uint8)) / len_pixel
    S = (S < atom_r) * 1

    return S

@curry
def generator_fft(indxs, atom_r, box_dim, len_pixel, full=False, scaler=0.0):

    S = np.zeros(box_dim)
    S[indxs[0], indxs[1], indxs[2]] = 1

    if full:
        scaled_box_dim = box_dim + scaler
        S = padder(S, scaled_box_dim, 0)
    else:
        scaled_box_dim = box_dim

    atom_r = padder(atom_r, scaled_box_dim)
    S = (imfilter(S, atom_r) > 1e-1) * 1

    return S
