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

>>> from fmks.fext import pipe, iterate_times

>>> np.random.seed(99)
>>> solve = solve_cahn_hilliard(gamma=4.)

>>> def tester(data, min_, max_, steps):
...     return pipe(
...         data,
...         iterate_times(solve, steps),
...         lambda x: x.flatten(),
...         lambda x: max(x) > max_ and min(x) < min_
...     )

1D

>>> assert tester(0.01 * (2 * np.random.random((2, 100)) - 1),
...              -2e-3, 2e-3, 10000)

2D

>>> assert tester(0.01 * (2 * np.random.random((2, 101, 101)) - 1),
...               -0.001, 0.001, 100)

3D

>>> assert tester(0.01 * (2 * np.random.random((2, 101, 101, 101)) - 1),
...               -0.0005, 0.0005, 10)


"""

import numpy as np
from fmks.fext import ifftn, fftn, curry, pipe, juxt, identity, iterate_times

def _k_space(size, dx):
    size1 = lambda: (size // 2) if (size % 2 == 0) else (size - 1) // 2
    size2 = lambda: size1() if (size % 2 == 0) else size1() + 1
    k = lambda: np.concatenate((
        np.arange(size)[:size2()],
        (np.arange(size) - size1())[:size1()]
    ))
    return k() * 2 * np.pi / (dx * size)

def _calc_ksq(x_data, dx):
    i_ = lambda: np.indices(x_data.shape[1:])
    return np.sum(_k_space(x_data.shape[1], dx)[i_()] ** 2, axis=0)[None]

def _axes(x_data):
    return np.arange(len(x_data.shape) - 1) + 1

def _f_response(x_data, dt, gamma, ksq, a1=3., a2=0.):
    FX = lambda: fftn(x_data, axes=_axes(x_data))
    FX3 = lambda: fftn(x_data ** 3, axes=_axes(x_data))
    explicit = lambda: a1 - gamma * a2 * ksq
    implicit = lambda: (1 - gamma * ksq) - explicit()
    dt_ = lambda: dt * ksq
    return (FX() * (1 + dt_() * explicit()) - dt_() * FX3()) / (1 - dt_() * implicit())


@curry
def solve_cahn_hilliard(x_data, dx=0.25, dt=0.001, gamma=4.):
    return ifftn(_f_response(_check(x_data),
                             dt,
                             gamma,
                             _calc_ksq(x_data, dx)),
                 axes=_axes(x_data)).real


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

def generate_cahn_hilliard_data(n_sample, size, dx=0.25, width=1., dt=0.001, n_steps=1):
    """Generate microstructures and responses for Cahn-Hilliard.

    Interface to generate random concentration fields and their
    evolution to be used for the fit method in the localizatoin
    regression model.

    Args:
      n_sample: number of microstructure samples
      size: size of the microstructure
      dx: grid spacing
      dt: time step size
      width: interface width between phases.
      n_steps: number of time steps used

    Returns:
      Tuple containing the microstructures and responses.

    Example

    >>> X, y = generate_cahn_hilliard_data(1, (6, 6))

    """
    solve = solve_cahn_hilliard(dx=dx, dt=dt, gamma=width**2)
    return pipe(
        2 * np.random.random((n_sample,) + size) - 1,
        juxt(identity, iterate_times(solve, n_steps))
    )
