import numpy as np
from fmks.data.cahn_hilliard import solve_cahn_hilliard
import dask.array as da

from fmks.fext import pipe, iterate_times

np.random.seed(99)
solve = solve_cahn_hilliard(gamma=4.)

def tester(data, min_, max_, steps):
    return pipe(
        data,
        iterate_times(solve, steps),
        # lambda x: x.flatten(),
        # lambda x: max(x) > max_ and min(x) < min_
    )

import dask
import time
t0 = time.time()
data = 0.01 * (2 * da.random.random((2, 100), chunks=(2, 100)) - 1)
for _ in range(100):
    for _ in range(100):
        data = solve(data)
    dask.to_hdf
    # npdata = data.compute()
    # data = da.from_array(npdata, chunks=(2, 100))

print(data.compute())
print(time.time() - t0)
# print(tester(0.01 * (2 * da.random.random((2, 100), chunks=(2, 100)) - 1),
#              -2e-3, 2e-3, 10000).compute())


>>> import h5py
>>> f = h5py.File('myfile.hdf5')
>>> dset = f['/data/path']

>>> import dask.array as da
>>> x = da.from_array(dset, chunks=(1000, 1000))

>>> da.to_hdf5('myfile.hdf5', '/y', y)  # doctest: +SKIP
