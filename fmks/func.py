"""Functional helper functions for fmks.
"""

import tempfile
from functools import wraps
from typing import Callable, Tuple, Any, List

import numpy as np
import dask.array as da
import h5py
import toolz.curried
from toolz.curried import iterate


def curry(func: Callable) -> Callable:
    """Curry a function, but keep the docstring

    Args:
      func: a function to curry

    Returns:
      a curried function

    >>> def add(a, b):
    ...     '''Add stuff'''
    ...     return a + b
    >>> curry(add)(1)(2)
    3
    >>> print(curry(add).__doc__)
    Add stuff
    """
    return wraps(func)(toolz.curried.curry(func))


@curry
def array_from_tuple(data: List[Tuple[slice, Any]],
                     shape: Tuple[int, ...],
                     dtype: np.dtype) -> np.ndarray:
    """Create an array from a list of slices and values.

    Args:
      data: list of slices and values used to create a new array
      shape: the shape of the new array
      dtype: the type of the new array


    Returns:
      the new populated array

    >>> array_from_tuple(
    ...     (((slice(None), slice(2)), 3.), ((2, 2), -1)),
    ...     (3, 3),
    ...     int
    ... )
    array([[ 3,  3,  0],
           [ 3,  3,  0],
           [ 3,  3, -1]])
    """
    arr = np.zeros(shape, dtype=dtype)
    for slice_, value in data:
        arr[slice_] = value
    return arr


@curry
def iterate_times(func: Callable, times: int, value: Any):
    """Iterate a function over a value.

    Args:
      func: the function to iterate
      times: the number of times to iterate
      value: the value to iterate over

    Returns:
      an update value

    >>> def inc(value):
    ...     return value + 1

    >>> iterate_times(inc, 0, 1)
    1

    >>> iterate_times(inc, 3, 1)
    4
    """
    iter_ = iterate(func, value)
    for _ in range(times):
        next(iter_)
    return next(iter_)


def write_read(data, datapath='/data'):
    """Write Dask array to disk and then read it.

    Required to force Dask array computation without using compute.

    Args:
      data: the Dask array to be written
      datapath: where to save the data in the HDF file

    Returns:
      the read in Dask array

    >>> print(write_read(da.arange(10, chunks=(2,))))
    dask.array<array, shape=(10,), dtype=int64, chunksize=(2,)>

    """
    with tempfile.NamedTemporaryFile(suffix='.hdf5') as file_:
        da.to_hdf5(file_.name, datapath, data)
        return da.from_array(h5py.File(file_.name)[datapath],
                             chunks=data.chunks)


@curry
def da_iterate(func, times, data, evaluate=100):
    """Iterate a Dask array workflow.

    Iterating a Dask array worflow requires periodic evaluation of the
    graph to ensure that the graph does not become too large. The
    graph is evaluated by the number steps indicated by `evaluate`.

    Args:
      func: the function to call at every iteration
      times: the number of iterations
      data: a Dask array to interate over

    Returns:
      the iterated data

    >>> iter_ = da_iterate(lambda x: x + 1)
    >>> print(iter_(3, da.arange(4, chunks=(2,))).compute())
    [3 4 5 6]

    >>> print(iter_(103, da.arange(4, chunks=(2,))).compute())
    [103 104 105 106]

    >>> print(iter_(0, da.arange(4, chunks=(2,))).compute())
    [0 1 2 3]
    """
    for _ in range(times // evaluate):
        data = write_read(iterate_times(func, evaluate, data))
    return iterate_times(func, times % evaluate, data)


@curry
def map_blocks(func, data):
    """Curried version of Dask's map_blocks

    Args:
      func: the function to map
      data: a Dask array

    Returns:
      a new Dask array

    >>> f = map_blocks(lambda x: x + 1)
    >>> f(da.arange(4, chunks=(2,)))
    dask.array<lambda, shape=(4,), dtype=int64, chunksize=(2,)>
    """
    return da.map_blocks(func, data)


allclose = curry(np.allclose)  # pylint: disable=invalid-name

fft = curry(np.fft.fft)  # pylint: disable=invalid-name

ifft = curry(np.fft.ifft)  # pylint: disable=invalid-name

fftn = curry(np.fft.fftn)  # pylint: disable=invalid-name

rfftn = curry(np.fft.rfftn)  # pylint: disable=invalid-name

ifftn = curry(np.fft.ifftn)  # pylint: disable=invalid-name

irfftn = curry(np.fft.irfftn)  # pylint: disable=invalid-name

fftshift = curry(np.fft.fftshift)  # pylint: disable=invalid-name

daifftn = curry(da.fft.ifftn)  # pylint: disable=invalid-name

dafftn = curry(da.fft.fftn)  # pylint: disable=invalid-name
