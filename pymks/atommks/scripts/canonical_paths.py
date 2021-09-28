import glob
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
from itertools import product, combinations
from toolz.curried import pipe, curry, compose
from tqdm.contrib.concurrent import process_map

def loader(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def saver(fname, obj):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)
        
        
def dtw_distance(x,y):
    from fastdtw import fastdtw
    return fastdtw(x, y, dist=1)[0]


def calc_path_distances_matrix(paths1, paths2=None, n_workers=1):
    if paths2 is not  None:
        out = process_map(lambda t: dtw_distance(*t), list(product(paths1, paths2)), max_workers=n_workers)
        return np.reshape(out, (len(paths1), len(paths2)))
    else:
        l = len(paths1)
        out = process_map(lambda t: dtw_distance(*t), 
                          list(combinations(paths1, r=2)), 
                          max_workers=n_workers)
        mat = np.zeros([l]*2)
        for i0, (ix, iy) in enumerate(combinations(range(l), r=2)):    
            mat[ix, iy] = out[i0]
            mat[iy, ix] = out[i0]
        return mat

    
def dmatrix(paths1, paths2=None, func=dtw_distance):
    
    if paths2 is not  None:
        mat = np.zeros((len(paths1), len(paths2)))
        for i1, p1 in enumerate(paths1):
            for i2, p2 in enumerate(paths2):
                mat[i1, i2]= func(p1, p2)
    else:
        mat = np.zeros([len(paths1)]*2)
        for i1, p1 in enumerate(paths1):
            for i2, p2 in enumerate(paths1[i1:]):
                mat[i1, i1+i2]= func(p1, p2)     
        out = mat.T + mat
        mask = np.eye(out.shape[0],dtype=bool)
        out[mask] = mat[mask]
        mat = out
    return mat


def get_canonical_paths(paths):
    
    from sklearn.cluster import AgglomerativeClustering
    
    canonical_paths = []
    
    if len(paths) > 1:
        
        data_matrix = dmatrix(paths, func=dtw_distance)
        model = AgglomerativeClustering(affinity='precomputed', 
                                        n_clusters=None, 
                                        linkage='complete', 
                                        distance_threshold=10.).fit(data_matrix)
        
        found = set()
        
        for i, l in enumerate(model.labels_):
            if l not in found:
                found.add(l)
                canonical_paths.append(paths[i])
    else:
        canonical_paths.append(paths[0])
        
    return canonical_paths


  


@curry
def get_canonical_paths_prll(paths, n_workers=1):
    
    from itertools import product
    from sklearn.cluster import AgglomerativeClustering
    
    canonical_paths = []
    
    if len(paths) > 1:
        
        out = process_map(dtw_distance_tuple, 
                          list(product(paths, paths)), 
                          max_workers=n_workers)
        
        data_matrix = np.reshape(out, (len(paths), len(paths)))
        
        model = AgglomerativeClustering(affinity='precomputed', 
                                        n_clusters=None, 
                                        linkage='complete', 
                                        distance_threshold=10.).fit(data_matrix)
        
        found = set()
        
        for i, l in enumerate(model.labels_):
            if l not in found:
                found.add(l)
                canonical_paths.append(paths[i])
    else:
        canonical_paths.append(paths[0])
        
    return canonical_paths


if __name__=="__main__":
    
    for cif_ix, cif in enumerate(cif_shortlist):

        fname = flist[ciflist.index(cif)]

        path = loader(fname)

        dists = path["dist_list"]

        torts = path["torts"]

        dists_dict = defaultdict(list)
        for ix, t in enumerate(torts):
            dists_dict[int(np.ceil(t*10))].append(dists[ix])
        dists_dict = OrderedDict(sorted(dists_dict.items()))

        print(cif_ix, fname, len(dists))



        strt = time.time()
        if cif not in paths_canonical:

            if len(dists) < 50000:

                out = process_map(get_canonical_paths, 
                                  [dists_dict[k] for k in dists_dict], 
                                  max_workers=14)
                paths_canonical[cif] = get_canonical_paths_prll(n_workers=14)(list(itertools.chain(*out))) 

                print(f"elpsd: {time.time()-strt} s, {len(paths_canonical[cif])}")

    #             saver(f"paths_canonical_LEVff-[1,1,0]-L-0.235821_0-U-0.758686_0-ss-19.309257903.pkl", paths_canonical)
        else:
            print(f"elpsd: {time.time()-strt} s, skipped, {cif}")
    
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