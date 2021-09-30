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
        # return process_map(func, inputs_list, max_workers=n_workers)


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


@curry
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
                canonical_paths.append(paths[i])
    else:
        canonical_paths.append(paths[0])
    return canonical_paths


if __name__=="__main__":
    import torch
    # from multiprocessing import Pool
    # from os import getpid

    # def double(i):
    #     print("I'm process", getpid())
    #     return i * 2

    # with Pool(4) as pool:
    #     result = pool.map(double, [1, 2, 3, 4, 5])
    #     print(result)

    # paths = torch.load("paths.pth")
    # inputs_list = list(combinations(paths[:10], r=2))
    # def func(t):
    #     return calc_path_distance(*t)
    # with Pool(4) as pool:
    #     result = pool.map(func, inputs_list)

    # import pdb; pdb.set_trace()


    # for cif_ix, cif in enumerate(cif_shortlist):

    #     fname = flist[ciflist.index(cif)]

    #     path = loader(fname)

    #     dists = path["dist_list"]

    #     torts = path["torts"]

    #     dists_dict = defaultdict(list)
    #     for ix, t in enumerate(torts):
    #         dists_dict[int(np.ceil(t*10))].append(dists[ix])
    #     dists_dict = OrderedDict(sorted(dists_dict.items()))

    #     print(cif_ix, fname, len(dists))



    #     strt = time.time()
    #     if cif not in paths_canonical:

    #         if len(dists) < 50000:

    #             out = process_map(get_canonical_paths, 
    #                               [dists_dict[k] for k in dists_dict], 
    #                               max_workers=14)
    #             paths_canonical[cif] = get_canonical_paths_prll(n_workers=14)(list(itertools.chain(*out))) 

    #             print(f"elpsd: {time.time()-strt} s, {len(paths_canonical[cif])}")

    # #             saver(f"paths_canonical_LEVff-[1,1,0]-L-0.235821_0-U-0.758686_0-ss-19.309257903.pkl", paths_canonical)
    #     else:
    #         print(f"elpsd: {time.time()-strt} s, skipped, {cif}")
    
#     flist = sorted(glob.glob("likely-min/*.pkl"))
    
#     ciflist = [f.split("/")[-1].split("_paths")[0].split("zz_")[-1] for f in flist]
    
#     paths_canonical = {}

#     for idx, cif in enumerate(tqdm(ciflist)):

#         path = loader(flist[idx])

#         dists = path["dist_list"]

#         torts = path["torts"]

#         dists_dict = defaultdict(list)

#         for ix, t in enumerate(torts):
#             dists_dict[int(np.ceil(t*10))].append(dists[ix])

#         paths_all = [dists_dict[k] for k in dists_dict]

#         with Pool(14) as P:
#             out = P.map(get_canonical_paths, paths_all)    
#         paths_canonical[cif] = get_canonical_paths(list(itertools.chain(*out)))
        
#         saver("likely_min_canonical_paths.pkl", paths_canonical)