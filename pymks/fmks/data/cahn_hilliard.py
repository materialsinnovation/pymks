r"""Solve the `Cahn-Hilliard equation to generate data.

Solve the Cahn-Hilliard equation,
<https://en.wikipedia.org/wiki/Cahn-Hilliard_equation>`__, for
multiple samples in arbitrary dimensions. The concentration varies
from -1 to 1. The equation is given by

.. math::

   \dot{\phi} = \nabla^2 \left( \phi^3 -
                                \phi \right) - \gamma \nabla^4 \phi

The discretiztion scheme used here is from `Chang and Rutenberg
<http://dx.doi.org/10.1103/PhysRevE.72.055701>`__. The scheme is a
semi-implicit discretization in time and is given by

.. math::

   \phi_{t+\Delta t}
   + \left(1 - a_1\right) \Delta t \nabla^2 \phi_{t+\Delta t}
   + \left(1 - a_2\right) \Delta t \gamma \nabla^4 \phi_{t+\Delta t}
   = \phi_t
   - \Delta t \nabla^2 \left(a_1 \phi_t + a_2
                             \gamma \nabla^2 \phi_t - \phi_t^3 \right)

where :math:`a_1=3` and :math:`a_2=0`.

>>> import dask.array as da
>>> from toolz.curried import pipe

>>> solve_ = np_solve(gamma=1., delta_t=1.)

>>> def tester(shape, min_, max_, steps):
...     return pipe(
...         0.01 * (2 * da.random.random(shape, chunks=shape) - 1),
...         map_blocks(iterate_times(solve_, steps)),
...         lambda x: da.max(x) > max_ and da.min(x) < min_
...     )

1D

>>> da.random.seed(101)
>>> assert tester((2, 100), -0.9, 0.9, 100)

2D

>>> da.random.seed(101)
>>> assert tester((2, 101, 101), -0.9, 0.9, 100)

3D

>>> da.random.seed(101)
>>> assert tester((2, 101, 101, 101), -5e-4, 5e-4, 10)

"""

import numpy as np
from toolz.curried import memoize
from ..func import curry, map_blocks, ifftn, fftn, iterate_times


def _k_space(size):
    size1 = lambda: (size // 2) if (size % 2 == 0) else (size - 1) // 2
    size2 = lambda: size1() if (size % 2 == 0) else size1() + 1
    return np.concatenate(
        (np.arange(size)[: size2()], (np.arange(size) - size1())[: size1()])
    )


@memoize
def _calc_ksq_(shape):
    indices = lambda: np.indices(shape)
    return np.sum(_k_space(shape[0])[indices()] ** 2, axis=0)[None]


def _calc_ksq(x_data, delta_x):
    return _calc_ksq_(x_data.shape[1:]) * (2 * np.pi / (delta_x * x_data.shape[1])) ** 2


def _axes(x_data):
    return np.arange(len(x_data.shape) - 1) + 1


def _explicit(gamma, ksq, param_a1=3.0, param_a2=0.0):
    return param_a1 - gamma * param_a2 * ksq


def _f_response(x_data, delta_t, gamma, ksq):
    fx_data = lambda: fftn(x_data, axes=_axes(x_data))
    fx3_data = lambda: fftn(x_data ** 3, axes=_axes(x_data))
    implicit = lambda: (1 - gamma * ksq) - _explicit(gamma, ksq)
    delta_t_ksq = lambda: delta_t * ksq
    numerator = (
        lambda: fx_data() * (1 + delta_t_ksq() * _explicit(gamma, ksq))
        - delta_t_ksq() * fx3_data()
    )
    return numerator() / (1 - delta_t_ksq() * implicit())


@curry
def np_solve(x_data, delta_x=0.25, delta_t=0.001, gamma=1.0):
    """Solve the Cahn-Hilliard equation for one step.

    Advance multiple microstuctures in time with the Cahn-Hilliard
    equation.

    Args:
      x_data: the initial microstucture
      delta_x: the grid spacing
      delta_t: the time step size
      gamma: Cahn-Hilliard parameter

    Returns:
      an updated microsturcture
n

    Raises:
      RuntimeError if domain is not square

    """
    return ifftn(
        _f_response(_check(x_data), delta_t, gamma, _calc_ksq(x_data, delta_x)),
        axes=_axes(x_data),
    ).real


def _check(x_data):
    """Ensure that domain is square.

    Args:
      x_data: the initial microstuctures

    Returns:
      the initial microstructures

    Raises:
      RuntimeError if microstructures are not square

    >>> _check(np.array([[[1, 2, 3], [4, 5, 6]]]))
    Traceback (most recent call last):
    ...
    RuntimeError: X must represent a square domain

    """
    if not np.all(np.array(x_data.shape[1:]) == x_data.shape[1]):
        raise RuntimeError("X must represent a square domain")

    return x_data


def solve_cahn_hilliard(x_data, n_steps=1, delta_x=0.25, delta_t=0.001, gamma=1.0):
    r"""Solve the Cahn-Hilliard equation.

    Solve the `Cahn-Hilliard equation
    <https://en.wikipedia.org/wiki/Cahn-Hilliard_equation>`_ for
    multiple samples in arbitrary dimensions. The concentration varies
    from -1 to 1. The equation is given by

    .. math::

       \dot{\phi} = \nabla^2 \left( \phi^3 - \phi \right) - \gamma \nabla^4 \phi

    The discretiztion scheme used here is from `Chang and Rutenberg
    <http://dx.doi.org/10.1103/PhysRevE.72.055701>`_. The scheme is a
    semi-implicit discretization in time and is given by

    .. math::

       \phi_{t+\Delta t}
       + \left(1 - a_1\right) \Delta t \nabla^2 \phi_{t+\Delta t}
       + \left(1 - a_2\right) \Delta t \gamma \nabla^4 \phi_{t+\Delta t}
       = \phi_t
       - \Delta t \nabla^2 \left(a_1 \phi_t + a_2
          \gamma \nabla^2 \phi_t - \phi_t^3 \right)

    where :math:`a_1=3` and :math:`a_2=0`.

    Args:
      x_data: dask array chunked along the sample axis ``(n_sample, n_x, n_y)``
      n_steps: number of time steps used
      delta_x: the grid spacing, :math:`\Delta x`
      delta_t: the time step size, :math:`\Delta t`
      gamma: Cahn-Hilliard parameter, :math:`\gamma`

    >>> import dask.array as da
    >>> da.random.seed(99)
    >>> x_data = 2 * da.random.random((1, 100, 100), chunks=(1, 100, 100)) - 1
    >>> y_data = solve_cahn_hilliard(x_data)
    >>> y_data.chunks
    ((1,), (100,), (100,))

    >>> y_data = solve_cahn_hilliard(x_data, n_steps=10000)  #doctest: +SKIP
    >>> from pymks import plot_microstructures
    >>> fig = plot_microstructures(x_data[0], y_data[0])
    >>> fig.show()  #doctest: +SKIP

    .. image:: cahn-hilliard.png
       :width: 400

    """
    return map_blocks(
        iterate_times(np_solve(delta_x=delta_x, delta_t=delta_t, gamma=gamma), n_steps),
        x_data,
    )
