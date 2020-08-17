"""Functions to generate a checkerboard microstructure
"""
from dask.array import from_array
from numpy import array, reshape, indices
from toolz.curried import pipe

from ..func import curry


@curry
def _checkerboard(size, square_shape):
    """Generte a checkerboard numpy array

    `square_shape` must be a numpy array that can divide into
    `np.indices(shape)`

    Args:
      size: the size of the domain
      square_shape: the shape of each subdomain

    """
    return pipe(size, indices, lambda x: x // square_shape, lambda x: x.sum(axis=0) % 2)


def generate_checkerboard(size, square_shape=(1,)):
    """Generate a 2-phase checkerboard microstructure

    Args:
      size: the size of the domain
      square_shape: the shape of each subdomain

    Returns:
      a microstructure of shape "(1,) + shape"

    >>> print(generate_checkerboard((4,)).compute())
    [[0 1 0 1]]
    >>> print(generate_checkerboard((3, 3)).compute())
    [[[0 1 0]
      [1 0 1]
      [0 1 0]]]
    >>> print(generate_checkerboard((3, 3), (2,)).compute())
    [[[0 0 1]
      [0 0 1]
      [1 1 0]]]
    >>> print(generate_checkerboard((5, 8), (2, 3)).compute())
    [[[0 0 0 1 1 1 0 0]
      [0 0 0 1 1 1 0 0]
      [1 1 1 0 0 0 1 1]
      [1 1 1 0 0 0 1 1]
      [0 0 0 1 1 1 0 0]]]
    """
    return pipe(
        square_shape,
        array,
        lambda x: reshape(x, x.shape + (1,) * len(size)),
        _checkerboard(size),
        lambda x: from_array(x[None], chunks=-1),
    )
