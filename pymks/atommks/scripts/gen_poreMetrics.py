import os
import ase
import time
import glob
import numpy as np
import ase.io as aio
import scipy.io as sio
import multiprocessing as mp
from toolz.curried import pipe, curry
import poremks.porosity as pore
import poremks.grid_generator as gen
from poremks.helpers import write2vtk

@curry
def poreStructureMaker(fname, save_dir="", r_probe=0.1, len_pixel=10):
    try:
        cif = pipe(fname,
                   lambda x: os.path.split(x)[-1].split("_dgrid")[0],
                   lambda x: os.path.join(save_dir, x))

        S = sio.loadmat(fname)["s"]

        pld = pore.get_pld(S)
        lcd = pore.get_lcd(S)

        # generates probe accessible pore region
        S_1 = (pore.gen_cleanPore(S,
                                  r_probe=r_probe,
                                  r_min=2.5,
                                  len_pixel=len_pixel) > 0) * 1

        # generates medial axis of the accessible pore region
        S_2 = pipe(S_1,
                   lambda x: np.pad(x,
                                    pad_width=((0,0),(0,0),(len_pixel, len_pixel)),
                                    mode = "constant", constant_values=1),
                   lambda x: pore.gen_medialAxis(x)[:,:,len_pixel:-len_pixel])

        # Prunes medial axis to return, only the paths connecting opposing surfaces
        S_3, paths = pore.gen_throughPath(S_2, depth=1)

        # Number of independant transport channels in the structure
        n_paths = len(pore.return_labelled(S_1)[-1])

        # accessible surface area
        asa = pore.get_asa(S_1, len_pixel=10)

        # accessile volume
        av = np.count_nonzero(S_1) * (1 / len_pixel)**3

        # pore size distribution
        psd = S[S_2==1]

        # dimensions of the structure
        dim = np.asarray(S.shape) / len_pixel

        # save all computed data as a matfile
        sio.savemat("%s_pore" % cif, {"pld":pld,
                                      "lcd":lcd,
                                      "n_paths":n_paths,
                                      "asa":asa,
                                      "av":av,
                                      "dim":dim,
                                      "paths":paths,
                                      "psd":psd,
                                      "len_pixel":len_pixel})

        print(cif, pld, lcd, asa, av, n_paths, np.mean(paths), np.mean(psd))

    except Exception as err:
        print("Exception for file : %s" % (fname), err)

def prll():
    input_folder = ""
    input_fnames = os.path.join(input_folder, "*_dgrid.mat")
    output_folder = ""

    flist = sorted(glob.glob())
    print("no. of files: %d" %len(flist))

    r_probe = 1.0 # Probe Radius
    len_pixel = 10 # No. of voxels per angstrom - resolution = 1/len_pixel

    func = poreStructureMaker(r_probe=r_probe,
                              len_pixel=len_pixel,
                              save_dir=output_folder)

    with mp.Pool(processes=2) as p:
        p.map(func, flist)

if __name__ == "__main__":
    start = time.time()
    prll()
    elpsd = time.time() - start
    print("time to complete: %1.3f s" % elpsd)
