"""Functional helper functions for fmks.
"""

from functools import wraps

import numpy as np
import dask.array as da
from dask import delayed
import toolz.curried
from toolz.curried import iterate, compose, pipe, get
from toolz.curried import map as map_


def curry(func):
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
def array_from_tuple(data, shape, dtype):
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
def iterate_times(func, times, value):
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


@curry
def map_blocks(func, data, chunks=None):
    """Curried version of Dask's map_blocks

    Args:
      func: the function to map
      data: a Dask array
      chunks: chunks for new array if reshaped

    Returns:
      a new Dask array

    >>> f = map_blocks(lambda x: x + 1)
    >>> f(da.arange(4, chunks=(2,)))
    dask.array<lambda, shape=(4,), dtype=int64, chunksize=(2,)>
    """
    return da.map_blocks(func, data, chunks=chunks)


allclose = curry(np.allclose)  # pylint: disable=invalid-name

fft = curry(np.fft.fft)  # pylint: disable=invalid-name

ifft = curry(np.fft.ifft)  # pylint: disable=invalid-name

fftn = curry(np.fft.fftn)  # pylint: disable=invalid-name

ifftn = curry(np.fft.ifftn)  # pylint: disable=invalid-name

fftshift = curry(np.fft.fftshift)  # pylint: disable=invalid-name

ifftshift = curry(np.fft.ifftshift)  # pylint: disable=invalid-name

daifftn = curry(da.fft.ifftn)  # pylint: disable=invalid-name

dafftn = curry(da.fft.fftn)  # pylint: disable=invalid-name

dafft = curry(da.fft.fft)  # pylint: disable=invalid-name

daifft = curry(da.fft.ifft)  # pylint: disable=invalid-name

dafftshift = curry(da.fft.fftshift)  # pylint: disable=invalid-name

daifftshift = curry(da.fft.ifftshift)  # pylint: disable=invalid-name

daconj = curry(da.conj)  # pylint: disable=invalid-name


def rcompose(*args):
    """Compose functions in order

    Args:
      args: the functions to compose

    Returns:
      composed functions

    >>> assert rcompose(lambda x: x + 1, lambda x: x * 2)(3) == 8
    """
    return compose(*args[::-1])


sequence = rcompose  # pylint: disable=invalid-name


def apply_dict_func(func, data, shape_dict):

    """Apply a function that returns a dictionary of arrays to a Dask
    array in parallel.

    e.g.

    >>> data = da.from_array(np.arange(36).reshape(4, 3, 3), chunks=(2, 3, 3))
    >>> def func(arr):
    ...     return dict(
    ...         a=np.resize(
    ...             arr,
    ...             (arr.shape[0],) + (arr.shape[1] + 1,) + (arr.shape[2] + 1,)
    ...         ),
    ...         b=np.resize(arr, (arr.shape[:3]) + (1,))
    ...     )
    >>> out = apply_dict_func(func, data, dict(a=(4, 4, 4), b=(4, 3, 3, 1)))
    >>> print(out['a'].chunks)
    ((2, 2), (4,), (4,))
    >>> print(out['b'].shape)
    (4, 3, 3, 1)
    >>> print(out['a'].compute().shape)
    (4, 4, 4)

    Args:
      func: the function to apply to the Dask array
      data: the Dask array to call the function with
      shape_dict: a dictionary of shapes, the keys are the keys
        returned by the funcion and the shapes correspond to the
        shapes that are output from the function

    """

    @curry
    def from_delayed(key, shape, delayed_func):
        return da.from_delayed(
            delayed_func, dtype=float, shape=(shape[0],) + shape_dict[key][1:]
        )

    def concat(key):
        return pipe(
            lambda x: func(np.array(x)),
            delayed,
            lambda x: map_(lambda y: (y.shape, x(y)), data.blocks),
            map_(lambda x: (x[0], get(key, x[1]))),
            map_(lambda x: from_delayed(key, x[0], x[1])),
            list,
            lambda x: da.concatenate(x, axis=0),
        )

    return pipe(shape_dict.keys(), map_(lambda x: (x, concat(x))), dict)


@curry
def debug(stmt, data):  # pragma: no cover
    """Helpful debug function
    """
    print(stmt)
    print(data)
    return data


def flatten(data):
    """Flatten data along all but the first axis

    Args:
      data: data to flatten

    Returns:
      the flattened data

    >>> data = np.arange(18).reshape((2, 3, 3))
    >>> flatten(data).shape
    (2, 9)
    """
    return data.reshape(data.shape[0], -1)


def make_da(func):
    """Decorator to allow functions that only take Dask arrays to take
    Numpy arrays.

    Args:
      func: the function to be decorated

    Returns:
      the decorated function

    >>> @make_da
    ... def my_func(arr):
    ...     return arr + 1

    >>> my_func(np.array([1, 1]))
    dask.array<add, shape=(2,), dtype=int64, chunksize=(2,)>

    """

    def wrapper(arr, *args, **kwargs):
        return func(da.from_array(arr, chunks=arr.shape), *args, **kwargs)

    return wrapper
