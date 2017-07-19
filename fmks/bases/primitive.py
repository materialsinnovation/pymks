r"""Primitive basis for microstructure discretization

Example.

Discretize the microstructure function into `n_states` local states
such that:

.. math::

   \frac{1}{\Delta x} \int_{H} \int_{s} \Lambda(h - l)
   m(h, x) dx dh = m[l, s]

where :math:`\Lambda` is the primitive basis (also called hat basis)
function

.. math::

   \Lambda (h - l) = max \Bigg (1-\Bigg |\frac{h(L - 1)}{H} -
   \frac{Hl}{L-1} \Bigg|, 0\Bigg)

A microstructure function discretized with this basis is subject to
the following constraint

.. math::

   \sum_{l=0}^L m[l, s] = 1

which is equivalent of saying that every location is filled with some
configuration of local states.

Here is an example with 3 discrete local states in a microstructure.

>>> x_data = np.array([[[1, 1, 0],
...                     [1, 0 ,2],
...                     [0, 1, 0]]])
>>> assert(x_data.shape == (1, 3, 3))

>>> x_test = np.array([[[[0, 1, 0],
...                      [0, 1, 0],
...                      [1, 0, 0]],
...                     [[0, 1, 0],
...                      [1, 0, 0],
...                      [0, 0, 1]],
...                     [[1, 0, 0],
...                      [0, 1, 0],
...                      [1, 0, 0]]]])

The when a microstructure is discretized, the different local states are
mapped into local state space, which results in an array of shape
`(n_samples, n_x, n_y, n_states)`, where `n_states=3` in this case.
For example, if a cell has a label of 2, its local state will be
`[0, 0, 1]`. The local state can only have values of 0 or 1.

>>> from toolz import pipe
>>> assert pipe(
...     x_data,
...     discretize(n_state=3, max_=2),
...     curry(np.allclose)(x_test))

"""

import dask.array as da
import numpy as np
from .func import curry


def discretize_nomax(data, states):
    """Helper function for primitive discretization.

    Args:
      data: the data to discretize
      states: a sequence of local states

    Returns:
      the unnormalized discetized microstructure

    Example:

    >>> discretize_nomax(np.array([[0, 1]]), np.array([0., 0.5, 1.0]))
    array([[[ 1.,  0., -1.],
            [-1.,  0.,  1.]]])

    >>> discretize_nomax(da.linspace(0, 1, 9, chunks=(3,)),
    ...             da.linspace(0, 1, 6, chunks=(2,)))
    dask.array<sub, shape=(9, 6), dtype=float64, chunksize=(3, 2)>

    """
    return 1 - (abs(data[..., None] - states)) / (states[1] - states[0])


# def minmax(data, min_, max_):
#     """Bound the values in an array by min_ and max_.

#     Args:
#       data: the data to
#       min_: min value


#     >>>

#     """
#     return da.minimum(da.maximum(data, min_), max_)


@curry
def discretize(x_data, n_state, min_=0.0, max_=1.0, chunks=()):
    """Primitive discretization of a microstructure.

    Args:
      x_data: the data to discrtize
      n_state: the number of local states
      min_: the minimum local state
      max_: the maximum local state

    Returns:
      the discretized microstructure

    >>> discretize(da.random.random((12, 9), chunks=(3, 9)),
    ...            3,
    ...            chunks=(1,)).chunks
    ((3, 3, 3, 3), (9,), (1, 1, 1))

    >>> discretize(np.array([[0, 1], [0.5, 0.5]]), 3, chunks=(1,)).chunks
    ((2,), (2,), (1, 1, 1))

    >>> discretize(np.array([[0, 1], [0.5, 0.5]]), 3, chunks=(1,)).compute()
    array([[[ 1.,  0.,  0.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 0.,  1.,  0.],
            [ 0.,  1.,  0.]]])
    """
    return da.maximum(
        discretize_nomax(da.clip(x_data, min_, max_),
                         da.linspace(min_,
                                     max_,
                                     n_state,
                                     chunks=chunks or (n_state,))),
        0
    )


def redundancy(ijk):
    """Used in localization to remove redundant slices

    Args:
      ijk: the current index

    Returns:
      the redundant slice, (slice(-1),) when no redundancies
    """
    if np.all(np.array(ijk) == 0):
        return (slice(None),)
    return (slice(-1),)


@curry
def primitive_basis(x_data, n_state, min_=0.0, max_=1.0, chunks=()):
    """Primitive discretization of a microstucture

    Args:
      x_data: the data to discrtize
      n_state: the number of local states
      min_: the minimum local state
      max_: the maximum local state

    Returns:
      a tuple, the first entry is the discretized data, other entries
      are functions required for localization
    """
    return (discretize(x_data, n_state, min_=min_, max_=max_, chunks=chunks),
            redundancy)
