# author: Apaar Shanker


import numba
import numpy as np
from scipy.spatial import cKDTree
from toolz.curried import pipe, curry
from ase.neighborlist import neighbor_list


fft = curry(np.fft.fft)

ifft = curry(np.fft.ifft)

fftn = curry(np.fft.fftn)

ifftn = curry(np.fft.ifftn)

fftshift = curry(np.fft.fftshift)

ifftshift = curry(np.fft.ifftshift)

conj = curry(np.conj)

func = curry(lambda x, y: conj(x) * fftn(y))

fabs = curry(lambda x: np.absolute(x))


@curry
def imfilter(x_data, f_data):
    """
    to convolve f_data over x_data
    """

    return pipe(f_data,
                ifftshift,
                fftn,
                lambda x: conj(x)*fftn(x_data),
                ifftn,
                fabs)

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
                # See the scaled_positions.py test.
                fractional[:, i] %= 1.0
                fractional[:, i] %= 1.0
    return fractional


@curry
def get_real_positions(coords, cell):
    """Get position in real space coordinates"""
    return np.dot(cell.T, coords.T).T


def sphere(r=10):
    """
    args: radius of the sphere

    returns: A 3D cubic matric of dim (2*r+1)^1
    """
    return pipe(2*r+1,
                lambda x: np.mgrid[:x,:x,:x],
                lambda xx: (xx[0]-r)**2 + (xx[1]-r)**2+(xx[2]-r)**2,
                lambda x: (x<r*r)*1)

def get_coords(atom):
    coord_list = []
    sym_list = np.asarray(sorted(atom.get_chemical_symbols()))
    syms = np.unique(sym_list)

    coords = atom.get_positions()

    for sym in syms:
        coord_list.append(coords[np.where(sym_list == sym)[0]])
    return coord_list


@curry
def padder(inp, shape, const_val=0):
    """
    args :  input matrix, new shape

    returns : matrix reshaped to given shape
    """
    ls = np.floor((shape - inp.shape) / 2).astype(int)
    hs = np.ceil((shape - inp.shape) / 2).astype(int)
    return np.pad(inp, ((ls[0], hs[0]), (ls[1], hs[1]), (ls[2], hs[2])), 'constant', constant_values=const_val)


@curry
def compute_rdf(atom, cutoff =5.0, nbins =1001):
    """
    returns RDF and bin_edges
    """
    N = len(atom)

    bins = np.linspace(0.0, cutoff + 2, nbins)

    i, j, d, D = neighbor_list('ijdD', atom, cutoff=cutoff, self_interaction=False)

    h, bin_edges = np.histogram(d, bins)

    rdf = h / N

    return rdf, bin_edges

@curry
def compute_rdf_subset(atoms, indexes, cutoff=6.0, nbins=201):
    """
    returns RDF computer over a subset of atoms and bin_edges
    """
    i, j, d, D = neighbor_list('ijdD', atoms, cutoff=6, self_interaction=False)
    dlist = []
    for idx in indexes:
        dlist.append(d[np.where(i == idx)[0]])
    dlist = np.concatenate(dlist, axis=0)
    bins = np.linspace(0.0, cutoff+2, nbins)

    N = len(indexes)
    h, bin_edges = np.histogram(dlist, bins)

    rdf = h / N

    return rdf, bin_edges

@curry
def compute_rdf_cross(atom, cutoff_=5.0, nbins_=501, sym1_ = "O", sym2_ = "Si"):
    """
    returns auto and cross-RDF for different spcies and bin_edges
    """
    syms = np.asarray(atom.get_chemical_symbols())
    idx_o = np.where(syms == sym1_)[0]
    idx_s = np.where(syms == sym2_)[0]

    bins = np.linspace(0.0, cutoff_ + 2, 501)

    i, j, d, D = neighbor_list('ijdD', atom, cutoff=cutoff_, self_interaction=False)

    rdf_list = []

    _auxset = set(idx_o)
    d_list = []
    for idx in _auxset:
        a = list(j[np.where(i == idx)[0]])
        b = list(d[np.where(i == idx)[0]])
        d_list = d_list + [b[a.index(x)] for x in a if x in _auxset]

    h, bin_edges = np.histogram(d_list, bins)

    rdf = h / len(idx_o)
    rdf_list.append(rdf)

    _auxset = set(idx_s)
    d_list = []
    for idx in _auxset:
        a = list(j[np.where(i == idx)[0]])
        b = list(d[np.where(i == idx)[0]])
        d_list = d_list + [b[a.index(x)] for x in a if x in _auxset]

    h, bin_edges = np.histogram(d_list, bins)

    rdf = h / len(idx_s)
    rdf_list.append(rdf)

    _auxset = set(idx_s)
    d_list = []
    for idx in idx_o:
        a = list(j[np.where(i == idx)[0]])
        b = list(d[np.where(i == idx)[0]])
        d_list = d_list + [b[a.index(x)] for x in a if x in _auxset]

    h, bin_edges = np.histogram(d_list, bins)

    rdf = h / len(idx_o)
    rdf_list.append(rdf)

    return rdf_list, bin_edges


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


def get_voxelizedStats(coords1, coords2=None, r_stat=5.0, len_pixel=10, cell=[[1,0,0],[0,1,0],[0,0,1]], pbc=[1,1,1]):

    center = cell.sum(axis=0)*0.5

    if coords2 is None:
        coords2 = coords1.copy()

    n_atoms1 = coords1.shape[0]
    n_atoms2 = coords2.shape[0]

    stat_coords = np.zeros([n_atoms1 * n_atoms2, 3])

    for indx, coord in enumerate(coords1):
        shift = center - coord
        stat_coords[indx * n_atoms2:(indx+1) * n_atoms2,:] = coords2 + shift

    indexes = pipe(stat_coords,
                   get_scaled_positions(cell=cell, pbc=pbc),
                   get_real_positions(cell=cell),
                   lambda x: x - center + r_stat,
                   lambda x: (np.round(x * len_pixel)).astype(int))

    shape = [int(r_stat * 2 * len_pixel + 1)] * 3

    box = pipe(shape,
               lambda shape: np.zeros(shape),
               lambda box: box_count(box, indexes, indexes.shape[0], shape),
               lambda box: box / n_atoms1)
    return box

@curry
def get_voxelizedStats_tree(coords, coords0, r_stat, len_pixel):
    tree = cKDTree(coords)

    shape = np.asarray([int(r_stat * 2 * len_pixel + 1)] * 3) # shape of the statistics box

    stat_coords = []
    count = 0
    for i, coord in enumerate(coords0[:]):
        indxs = tree.query_ball_point(coord, 1.8 * r_stat)
        stat_coords.append(coords[indxs] - coord)

    stat_coords = np.concatenate(stat_coords, axis=0) + r_stat

    indexes = (np.round(stat_coords * len_pixel)).astype(int)

    box = np.zeros(shape)
    N = indexes.shape[0]
    box = box_count(box, indexes, N, shape)
    box = box / len(coords0)

    return box


@curry
def write2vtk(matrix, fname="zeo.vtk"):
    sx, sy, sz = matrix.shape
    mx = np.max(matrix)
    mi = np.min(matrix)
    lines ='# vtk DataFile Version 2.0\nVolume example\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\nASPECT_RATIO 1 1 1\nORIGIN 0 0 0\nPOINT_DATA %d\nSCALARS matlab_scalars float 1\nLOOKUP_TABLE default\n'%(sx, sy, sz, matrix.size)
    with open(fname, 'w') as f:
        f.write(lines)
        for ix in range(sz):
            v = np.ravel(matrix[:,:,ix], order="f")
            v = ["%1.5f"%x for x in np.round(100 * v / mx)]
            line = " ".join(v)
            f.write(line+"\n")
