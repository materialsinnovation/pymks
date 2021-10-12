"""
@author: ashanker9@gatech.edu
"""
import numba
import scipy
import numpy as np
from tqdm import tqdm
from .helpers import *
from scipy.spatial import cKDTree
from toolz.curried import pipe, curry
from collections import defaultdict, OrderedDict

from scipy.ndimage import measurements
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import remove_small_objects
from skimage.morphology import skeletonize_3d as sklz

from .canonical_paths import calc_path_distance, calc_path_distances_matrix, calc_canonical_paths


erasure = curry(remove_small_objects)


def return_labelled(data):
    S_l, n_count = measurements.label(data)
    top = list(np.unique(S_l[:,:,0]))[1:]
    bot = list(np.unique(S_l[:,:,-1]))[1:]
    m = list(set(top).intersection(bot))
    return S_l, n_count, m


def is_connected(data):
    S_l, n_count, m = return_labelled(data)
    if len(m) is 0:
        return False
    else:
        return True

    
@curry
def calc_euclidean_distance(data, n_pixel=1, axis=-1):
    """Calculate the Euclidean distance from one phase to another

    Given a two phase microstructure labeled 1 and 0, calculate the
    distance of each of the 1-voxels from the nearest
    0-voxels. Returns 0 for 0-voxels and a float value indicating the
    distance at 1-voxels.

    Args:
      data: the two-phase microstructure (either 0 or 1) in any dimension
      n_pixel: number of pixels per unit length

    Works with only zeros

    >>> one_phase = np.array([[0, 0], [0, 0]])
    >>> calc_euclidean_distance(one_phase)
    array([[0., 0.],
           [0., 0.]])

    Simple test case

    >>> two_phase = np.ones((3, 3))
    >>> two_phase[1, 1] = 0
    >>> assert(np.allclose(
    ...     calc_euclidean_distance(two_phase),
    ...     [[np.sqrt(2), 1, np.sqrt(2)],
    ...      [1, 0, 1],
    ...      [np.sqrt(2), 1, np.sqrt(2)]]
    ... ))

    """
    
    
    if axis == -1 or axis == (data.ndim - 1):
        data = data
    else:
        data = np.rot90(data, axes=(axis, -1))

    data =  np.pad(
        data, 
        ((1, 1), (1, 1), (0, 0)), 
        'constant', 
        constant_values=0
    )
    
    return transform_edt(data.astype(np.uint8))[1:-1, 1:-1, :] / n_pixel


@curry
def calc_accessible_pore(dist=None, r_probe=0.5, r_min=2.5, n_pixel=10):
    """
    From distance grid matrix generate accessible pore region
    removing small, unconnected regions(r_min sized)
    """

    return pipe(dist,
               lambda s: (s > r_probe) * 1,
            #    lambda s: return_labelled(s)[0],
            #    erasure(min_size = 4/3 * np.pi * (r_min * n_pixel)**3),
    )


@curry
def get_pld(data, lo=0.5, hi=9.5, tol=0.1):
    """Calculate the pore length diameter (PLD).

    The PLD signifies the largest structure that can pass through a
    connected pore structure. The ``lo`` and ``hi`` values are guesses
    for the PLD to help the solution converge. The ``tol`` value is
    the tolerance to search to find the PLD. The maximum number of
    iterations is approximately ``(hi - lo) / tol``. Typically the
    number of iterations is less that this.

    Args:
      data: the Euclidean distance of the pore strcuture from nearest
        atoms.
      lo: the minimum PLD value to check for
      hi: the maximum PDL value to check for
      tol: the resolution of the PLD values

    Set up a test with a 6x6x6 array with a 2x2 tube through it in the
    z-direction. The value of PLD should be 2 in this case.

    >>> pore_data = np.zeros((6, 6, 6))
    >>> pore_data[2:4, 2:4, :] = 1
    >>> assert np.allclose(
    ...    get_pld(calc_euclidean_distance(pore_data), tol=0.01),
    ...    2.0,
    ...    atol=0.01
    ... )

    """
    computed = False
    mid = (lo + hi) * 0.5
    while not computed:
        S_mod = np.zeros(data.shape)
        S_mod[data >= mid] = 1
        if is_connected(S_mod):
            lo = mid
        else:
            hi = mid

        mid = (lo + hi) * 0.5
        if hi - lo < tol:
            computed = True
            pld = mid * 2

    return pld


def get_lcd(data):
    """Calculate the largest cavity distance (LCD).

    The LCD signifies the largest cavity in the pore structure (the
    radial size of the largest cavity).

    Args:
      data: the Euclidean distance of the pore strcuture from nearest
        atoms

    >>> data = np.array([[0,  4, 0], [1, 3, 1], [0, 1, 0]])
    >>> get_lcd(data)
    8

    """
    return 2 * data.max()


def get_asa(data, r_probe=0.5, n_pixel=10):
    """
    Calculate the accessible surface area (asa).
    """
    data = calc_accessible_pore(data, r_probe=r_probe, n_pixel=n_pixel)
    w = [[[1,1,1],[1,1,1],[1,1,1]],
         [[1,1,1],[1,1,1],[1,1,1]],
         [[1,1,1],[1,1,1],[1,1,1]]]
    data_blur = scipy.signal.fftconvolve(data, w, mode="same") > 1e-6
    return (np.count_nonzero(data_blur) - np.count_nonzero(data)) * (1 / n_pixel)**2


def get_av(data, r_probe=0.5, n_pixel=10):
    """
    Calculate the accessible volume (av)
    """
    data = calc_accessible_pore(data, r_probe=r_probe, n_pixel=n_pixel)
    return np.count_nonzero(data) * (1/n_pixel)**3


def calc_pore_metrics(dist, lo=0.5, hi=9.5, tol=0.1, axis=-1, r_probe=0.5, n_pixel=1):
    """Calulate the pore metrics.

    The pore metrics consist of the pore limiting diameter (PLD), the
    largest cavity diameter (LCD), the accessible surface area (ASA) 
    and the accessible volume (AV).

    The PLD signifies the largest structure that can pass through a
    connected pore structure. The ``lo`` and ``hi`` values are guesses
    for the PLD to help the solution converge. The ``tol`` value is
    the tolerance to search to find the PLD. The maximum number of
    iterations is approximately ``(hi - lo) / tol``. Typically the
    number of iterations is less that this.

    The LCD signifies the largest cavity in the pore structure (the
    radial size of the largest cavity).
    
    The ASA for is the combined internal surface area of all cavities that 
    can be accessed by a foreign molecule. 
    
    The AV is the combined volume of all cavities that can be accessed by 
    a foreign molecule.

    Args:
      data: the two-phase microstructure (either 0 or 1) in any dimension
      lo: the minimum PLD value to check for
      hi: the maximum PLD value to check for
      tol: the resolution of the PLD values
      axis: the traversal direction of the probe molecule
      n_pixel: number of pixels per unit length

    Returns:
      a tuple with the first item being the distance grid and the second item 
      being a dictionary with ``pld``, ``lcd``, ``asa`` and ``av`` keys

    >>> data = np.zeros((3, 3, 3))
    >>> data[1, 1] = 1
    >>> assert(np.allclose(
    ...     calc_pore_metrics(data, tol=0.01)[1]['pld'],
    ...     2.0,
    ...     atol=0.01
    ... ))

    """

    return dict(
        pld=get_pld(dist, lo, hi, tol),
        lcd=get_lcd(dist), 
        asa=get_asa(dist, r_probe=r_probe, n_pixel=n_pixel), 
        av=get_av(dist, r_probe=r_probe, n_pixel=n_pixel)
    )


@curry
def calc_medial_axis(data):
    return pipe(data,
                lambda s: s.astype(np.uint8),
                lambda s: sklz(s))


def calc_path_length(path, coords):
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
def calc_shortest_paths(S, depth):
    coords = np.concatenate([np.where(S>0)], axis=1).T
    bot = []
    top = []
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
    torts_list = []
    indxs_list = []    
    S_1 = np.zeros(S.shape, dtype=np.uint8)
    for index, idx0 in enumerate(top):
        pred0 = pred[index,:]
        for idx in bot:
            path = [idx,]
            path, done = pred_search(pred0, path, idx0)
            if done:
                indxs_list.append([coords[path][:,idim] for idim in range(3)])
                dx, dy, dz = indxs_list[-1]
                S_1[dx, dy, dz] += 1
                torts_list.append(calc_path_length(path, coords) / S.shape[2])
    return S_1, torts_list, indxs_list


def calc_diffusion_paths(dists, r_probe=0.5, n_pixel=10, n_workers=12):
    """
    Calulate the path metrics.
    """
    
    paths, torts_lst, indxs_lst = pipe(
        dists,
        lambda x: (x > r_probe) * 1,
        lambda x: np.pad(x, 
                        pad_width=((0, 0),(0, 0),(n_pixel, n_pixel)), 
                        mode = "constant", 
                        constant_values=1
                    ),  
        lambda x: calc_medial_axis(x)[:,:,n_pixel:-n_pixel], 
        calc_shortest_paths(depth=1)
    )
    
    dlist = [dists[indxs] for indxs in indxs_lst]
    plds = [dists[indxs].min()*2 for indxs in indxs_lst]    
      
    dists_dict = defaultdict(list)
    
    for ix, (p, t) in enumerate(zip(plds, torts_lst)):
        dists_dict[(int(np.ceil(p*n_pixel)), int(np.ceil(t*n_pixel)))].append({"psv": dlist[ix], "indxs":indxs_lst[ix], "pld": p, "tort":t})
    dists_dict = OrderedDict(sorted(dists_dict.items()))

    canonical_dists_dict = {}
    for ix, (k, p) in tqdm(enumerate(dists_dict.items())):
        if len(p) > 2500:
            canonical_dists_dict[k] = p
        else:
            canonical_dists_dict[k] = calc_canonical_paths(p, n_workers=n_workers)
    
    return dists_dict, canonical_dists_dict