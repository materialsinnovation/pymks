import glob
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
from multiprocessing import Pool
from collections import defaultdict
from itertools import product, combinations
from toolz.curried import pipe, curry, compose
from tqdm.contrib.concurrent import process_map
from sklearn.cluster import AgglomerativeClustering


def calc_path_distance(path1, path2):
    return fastdtw(path1, path2, dist=1)[0]


def calc_func_over_list(func, inputs_list, n_workers):
    if n_workers == 1:
        return [func(item) for item in inputs_list]
    else:
        with Pool(n_workers) as pool:
            out = pool.map(func, inputs_list)
        return out


def calc_path_distance_for_tuple(t):
    return calc_path_distance(t[0]["psv"], t[1]["psv"])


def calc_path_distances_matrix(paths1, paths2=None, n_workers=1):

    if paths2 is not  None:
        out = calc_func_over_list(func=calc_path_distance_for_tuple, 
                                  inputs_list=list(product(paths1, paths2)), 
                                  n_workers=n_workers)
        return np.reshape(out, (len(paths1), len(paths2)))
    else:
        l = len(paths1)
        out = calc_func_over_list(func=calc_path_distance_for_tuple, 
                                  inputs_list=list(combinations(paths1, r=2)), 
                                  n_workers=n_workers)
        mat = np.zeros([l]*2)
        for i0, (ix, iy) in enumerate(combinations(range(l), r=2)):    
            mat[ix, iy] = out[i0]
            mat[iy, ix] = out[i0]
        return mat


def calc_canonical_paths(paths, n_workers=1, threshold=10.):
    canonical_paths = []
    if len(paths) > 1:
        distance_matrix = calc_path_distances_matrix(paths,  n_workers=n_workers)
        model = AgglomerativeClustering(affinity='precomputed', 
                                        n_clusters=None, 
                                        linkage='complete', 
                                        distance_threshold=threshold).fit(distance_matrix)
        found = set()
        for i, l in enumerate(model.labels_):
            if l not in found:
                found.add(l)
                paths[i]["count"] = np.count_nonzero(model.labels_==l)
                canonical_paths.append(paths[i])
    else:
        paths[0]["count"] = 1
        canonical_paths.append(paths[0])
    return canonical_paths
