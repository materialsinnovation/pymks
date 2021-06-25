import os
import ase
import time
import glob
import numpy as np
import ase.io as aio
import scipy.io as sio
import multiprocessing as mp
import poremks.porosity as pore
from toolz.curried import pipe, curry
import poremks.grid_generator as gen
from poremks.helpers import write2vtk



@curry
def structure_maker(fname, radii={"Si":1.35, "O": 1.35}, len_pixel=10, rep=[1,1,1], save_dir=""):
    """
    saves the voxelized structure in matfile format
    """
    try:

        cif  = pipe(fname,
                    lambda x: os.path.split(x)[-1][:-4],
                    lambda x: os.path.join(save_dir, x))

        atom = aio.read(fname).repeat(rep)
        S = gen.grid_maker(atom,
                           len_pixel=10,
                           radii=radii,
                           full=False,
                           fft=False)[0]

        padval = ((1, 1), (1, 1), (0, 0))
        S_dgrid = pipe(S,
                       lambda s: np.pad(s, padval, 'constant', constant_values=0),
                       lambda s: pore.dgrid(s, len_pixel))
        sio.savemat("%s_dgrid.mat" % cif, {'s':S_dgrid})
        write2vtk(S, "%s_pore.vtk" % cif)
        print(cif)
    except Exception as err:
        print("Exception for file : %s" % (fname), err)

def prll():
    save_dir = ""
    flist = sorted(glob.glob("*.cif"))
    print("No. of files: %d" % len(flist))
    func = structure_maker(len_pixel=10, rep=[1,1,1], save_dir=save_dir)
    with mp.Pool(processes=1) as p:
        p.map(func, flist)


if __name__ == "__main__":
    strt = time.time()
    prll()
    end = time.time()
    print(end - strt)
