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

    Args:
      n_phases: number of phases
      shape: the shape of the microstructure
      chunks: how to chunk the sample axis

    Returns:
      a dask array of delta microstructures

    >>> a = generate_delta(5, (3, 4), chunks=(5,))
    >>> a.shape
    (20, 3, 4)
    >>> a.chunks
    ((5, 5, 5, 5), (3,), (4,))
    """
    return da.from_array(_npgenerate(n_phases, shape), chunks=(chunks or (-1,)) + shape)
