"""Generate a delta microstructure for training elastic PyMKS models.

"""

from numpy import arange, transpose, array, identity
import dask.array as da

from ..func import npresize, extend, assign, sequence, curry


@curry
def _npgenerate(n_phases, shape):
    return sequence(
        arange,
        npresize((n_phases, n_phases)),
        transpose,
        extend(shape),
        assign(
            arange(n_phases)[None, :],
            ((slice(None), slice(None)) + tuple(array(shape) // 2)),
        ),
        lambda x: x[~identity(n_phases, dtype=bool)],
    )(n_phases)


@curry
def generate_delta(n_phases, shape, chunks=()):
    """Generate a delta microstructure

    A delta microstructure has a 1 at the center and 0 everywhere else
    for each phase. This is used to calibrate linear elasticity models
    that only require delta microstructures for calibration.

    Args:
      n_phases (int): number of phases
      shape (tuple): the shape of the microstructure, ``(n_x, n_y)``
      chunks (tuple): how to chunk the sample axis ``(n_chunk,)``

    Returns:
      a dask array of delta microstructures

    If `n_phases=5` for example, this requires 20 microstructures as
    each phase pairing requies 2 microstructure arrays.

    >>> arr = generate_delta(5, (3, 4), chunks=(5,))
    >>> arr.shape
    (20, 3, 4)
    >>> arr.chunks
    ((5, 5, 5, 5), (3,), (4,))
    >>> print(arr[0].compute())
    [[0 0 0 0]
     [0 0 1 0]
     [0 0 0 0]]

    `generate_delta` requires at least 2 phases

    >>> arr = generate_delta(2, (3, 3))
    >>> arr.shape
    (2, 3, 3)
    >>> print(arr[0].compute())
    [[0 0 0]
     [0 1 0]
     [0 0 0]]

    """
    return da.from_array(_npgenerate(n_phases, shape), chunks=(chunks or (-1,)) + shape)
