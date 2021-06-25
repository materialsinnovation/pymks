"""
@author: ashanker9@gatech.edu
"""

import ase
import numba
import numpy as np
from toolz.curried import pipe, curry, compose

try:
    import MDAnalysis
except ImportError:
    print("For defect atom identification")
    print("pip install MDAnalysis")
    pass


@curry
def get_scaled_positions(coords, cell, pbc, wrap=True):
    """Get positions relative to unit cell i.e. fractional coordinates.
    If wrap is True, atoms outside the unit cell will be wrapped into
    the cell in those directions with periodic boundary conditions
    so that the scaled coordinates are between zero and one.
    """
    fractional = np.linalg.solve(cell.T,
                                 coords.T).T

    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                # Yes, we need to do it twice.
                # See the scaled_positions.py test in ase library.
                fractional[:, i] %= 1.0
                fractional[:, i] %= 1.0
    return fractional


@curry
def get_real_positions(coords, cell):
    """Get position in real space coordinates"""
    return np.dot(cell.T, coords.T).T


@curry
def get_kdTree(coords, cell_dim, cutoff):
    import MDAnalysis
    
    tree = MDAnalysis.lib.pkdtree.PeriodicKDTree(box=cell_dim.astype(np.float32))
    tree.set_coords(coords.astype(np.float32), 
                    cutoff=np.float32(cutoff))
    
    return tree


@curry
def get_realStats(coords_all, coords_sub, r_stat, cutoff, cell, cell_dim, pbc,):
    import MDAnalysis
    
    tree = MDAnalysis.lib.pkdtree.PeriodicKDTree(box=cell_dim.astype(np.float32))
    tree.set_coords(coords_all.astype(np.float32), 
                    cutoff=np.float32(cutoff))

    frac_coords = get_scaled_positions(cell=cell, pbc=pbc, wrap=True)
    real_coords = get_real_positions(cell=cell)
    rescale = compose(real_coords, frac_coords)
    
    return pipe(tree.search_tree(coords_sub, radius=r_stat), 
                lambda indxs: rescale(coords_all[indxs[:,1]] - coords_sub[indxs[:,0]] + cell.diagonal()/2), 
                lambda crds: crds - cell.diagonal()/2)


@curry
def get_rdf(coords_stat, r_stat, len_pixel):
    
    get_dists = lambda c: np.sqrt(np.sum(c**2, axis=1))
    
    nbins = np.round(r_stat*len_pixel).astype(int)+1
    bins = np.linspace(0.0, r_stat+2, num=nbins)
    
    dists = get_dists(coords_stat)
    rdf, bin_edges = np.histogram(dists, bins)

    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    
    rdf = rdf / rdf[0]
    rdf[0] = 0.
    vols = 4 / 3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    pdf = rdf/vols
    
    return rdf, pdf, bin_centers


@curry
def get_2ptStat(coords_stat, r_stat, len_pixel):
    
    get_voxel_ids = lambda c, l: (np.round(c * l)).astype(int)
    
    coords_indx = get_voxel_ids(coords_stat + r_stat, len_pixel)
    shape = np.asarray([np.round(r_stat * 2 * len_pixel).astype(int) + 1] * 3)
    box = box_count(np.zeros(shape), 
                     coords_indx, 
                     len(coords_indx), shape)
  
    return box / box.max()


@numba.njit(parallel=True)
def box_count(box, indexes, N, shape):
    sx, sy, sz = shape
    for i in range(N):
        cx, cy, cz = indexes[i]
        if (cx < sx) and (cx >= 0):
            if (cy < sy) and (cy >= 0):
                if (cz < sz) and (cz >= 0):
                    box[cx, cy, cz] += 1
    return box