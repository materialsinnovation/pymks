"""
@author: ashanker9@gatech.edu
"""
import numba
import scipy
import numpy as np
from .helpers import *
from scipy.spatial import cKDTree
from toolz.curried import pipe, curry
from scipy.ndimage import measurements
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import remove_small_objects
from skimage.morphology import skeletonize_3d as sklz

erasure = curry(remove_small_objects)

@curry
def accessibleRegion(S, atomH, r_h, overlap=0.01):
    vol = np.count_nonzero(atomH)
    S_mod = imfilter(x_data=(S < 1e-6),
                     f_data = padder(atomH, np.array(S.shape))) / vol
    S_mod = S_mod < overlap
    return S_mod


@curry
def return_labelled(x_data):
    S_l, n_count = measurements.label(x_data)
    top = list(np.unique(S_l[:,:,0]))[1:]
    bot = list(np.unique(S_l[:,:,-1]))[1:]
    m = list(set(top).intersection(bot))
    return S_l, n_count, m


@curry
def is_connected(S):
        S_l, n_count, m = return_labelled(S)
        if len(m) is 0:
            return False
        else:
            return True

@curry
def get_pld(s, lo=0.5, hi=9.5, tol=0.1):
    """
    returns PLD
    input: EDT of the porous structure, and low and high range for PLD
    """
    computed = False
    mid = (lo + hi) * 0.5
    while not computed:
        S_mod = np.zeros(s.shape)
        S_mod[s >= mid] = 1
        if is_connected(S_mod):
            lo = mid
        else:
            hi = mid

        mid = (lo + hi) * 0.5
        if hi - lo < tol:
            computed = True
            pld = mid * 2

    return pld


@curry
def get_lcd(s, len_pixel=10):
    """
    returns LCD
    input: EDT of the porous structure
    """
    return 2 * s.max()

@curry
def dgrid(s, len_pixel):
    """
        args:
        s: 3D volume
        len_pixel: resolution
    """
    return transform_edt(s.astype(np.uint8)) / len_pixel



def get_asa(S, len_pixel=10):
    S = (S > 0) * 1
    w = [[[1,1,1],[1,1,1],[1,1,1]],
         [[1,1,1],[1,1,1],[1,1,1]],
         [[1,1,1],[1,1,1],[1,1,1]]]
    S1 = scipy.signal.fftconvolve(S, w, mode="same") > 1e-6
    return (np.count_nonzero(S1) - np.count_nonzero(S)) * (1 / len_pixel)**2


@curry
def gen_cleanPore(S=None, r_probe=0.5, r_min=2.5, len_pixel=10):
    """From distance grid matrix generate accessible pore region
    removing small, unconnected regions(r_min sized)"""

    erase = erasure(min_size = 4/3 * np.pi * (r_min * len_pixel)**3)
    S_1 = pipe(S,
               lambda s: (s > r_probe) * 1,
               lambda s: return_labelled(s)[0],
               erase,)
    return S_1


@curry
def gen_medialAxis(S):
    return pipe(S,
                lambda s: s.astype(np.uint8),
                lambda s: sklz(s))

def get_pathLength(path, coords):
    l = 0.0
    coord0 = coords[path[0]]
    for i, idx in enumerate(path[1:]):
        coord = coords[idx]
        l = l + np.sqrt(np.sum((coord - coord0)**2))
        coord0 = coord
    return l

@numba.njit
def pred_search(pred, path, idx0):
    done = False
    idx = path[0]
    while not done:
        idx = pred[idx]

        if idx == -9999:
            break
        path.append(idx)
        if idx == idx0:
            done = True
    return path, done

@curry
def gen_throughPath(S, depth):
    coords = np.concatenate([np.where(S>0)], axis=1).T
    i=0
    bot = []
    top = []

    tort = []
    l = S.shape[2]

    S_1 = np.zeros(S.shape, dtype=np.uint8)

    for i in range(depth):
        bot = bot + list(np.where(coords[:,2] == i)[0])
        top = top + list(np.where(coords[:,2] == (S.shape[2]-1-i))[0])

    tree = cKDTree(coords)
    dok_mat = tree.sparse_distance_matrix(tree, max_distance=2, p=2.0)
    dist_mat, pred = dijkstra(dok_mat,
                              directed=False,
                              indices=top,
                              return_predecessors=True,
                              unweighted=True)

    for index, idx0 in enumerate(top):
        pred0 = pred[index,:]
        for idx in bot:
            path = [idx,]

            path, done = pred_search(pred0, path, idx0)

            if done:
                dx, dy, dz = [coords[path][:,idim] for idim in range(3)]
                S_1[dx, dy, dz] += 1
                tort.append(get_pathLength(path, coords)/ l)

    return (S_1 > 0)*1, tort
