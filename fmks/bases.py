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

>>> from fmks.fext import pipe
>>> assert pipe(
...     x_data,
...     discretize(n_state=3, max_=2),
...     curry(np.allclose)(x_test))

"""

from typing import Callable, Tuple

import numpy as np
from .fext import curry


def _discretize(x_data: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Helper function for primitive discretization.

    Args:
      x_data: the data to discretize
      states: a sequence of local states

    Returns:
      the unnormalized discetized microstructure

    Example:

    >>> _discretize(np.array([[0, 1]]), np.array([0., 0.5, 1.0]))
    array([[[ 1.,  0., -1.],
            [-1.,  0.,  1.]]])
    """
    return 1 - (abs(x_data[..., None] - states)) / (states[1] - states[0])


def _minmax(data, min_, max_):
    return np.minimum(np.maximum(data, min_), max_)


@curry
def discretize(x_data: np.ndarray,
               n_state: int,
               min_: float = 0.0,
               max_: float = 1.0) -> np.ndarray:
    """Primitive discretization of a microstructure.

    Args:
      x_data: the data to discrtize
      n_state: the number of local states
      min_: the minimum local state
      max_: the maximum local state

    Returns:
      the discretized microstructure
    """
    return np.maximum(
        _discretize(_minmax(x_data, min_, max_),
                    np.linspace(min_, max_, n_state)),
        0
    )


def redundancy(ijk: tuple) -> tuple:
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
def primitive_basis(x_data: np.ndarray,
                    n_state: int,
                    min_: float = 0.0,
                    max_: float = 1.0) -> Tuple[np.ndarray, Callable]:
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
    return (discretize(x_data, n_state, min_=min_, max_=max_),
            redundancy)
