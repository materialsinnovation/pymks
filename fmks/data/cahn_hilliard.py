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

In 1D.

>>> np.random.seed(99)
>>> solve = solve_cahn_hilliard(gamma=4.)
>>> assert pipe(
...     0.1 * (2 * np.random.random((2, 100)) - 1),
...     iterate_times(solve, 10000),
...     lambda phi: max(phi.flat) > 2e-3) and (min(phi.flat) < -2e-3
... )

# In 2D.

# >>> N = 101
# >>> phi = 0.01 * (2 * np.random.random((2, N, N)) - 1)
# >>> ch = CahnHilliardSimulation(gamma=4.)
# >>> for i in range(100):
# ...     ch.run(phi)
# ...     phi[:] = ch.response
# >>> assert (max(phi.flat) > 0.001) and (min(phi.flat) < -0.001)

# In 3D.

# >>> phi = 0.01 * (2 * np.random.random((2, N, N, N)) - 1)
# >>> ch = CahnHilliardSimulation(gamma=4.)
# >>> for i in range(10):
# ...     ch.run(phi)
# ...     phi[:] = ch.response

# >>> assert (max(phi.flat) > 0.0005) and (min(phi.flat) < -0.0005)

"""

import numpy as np
from fmks.fext import ifftn, fftn, curry

def _k_space(size, dx):
    size1 = lambda: (size / 2) if (size % 2 == 0) else (size - 1) / 2
    size2 = lambda: size1() if (size % 2 == 0) else size1() + 1
    k = lambda: np.concatenate(np.arange(size)[:size2()],
                               (np.arange(size) - size1())[:size1()])
    return k() * 2 * np.pi / (dx * size)

def _calc_ksq(x_data, dx):
    i_ = lambda: np.indices(x_data.shape[1:])
    return np.sum(_k_space(x_data.shape[1], dx)[i_()] ** 2, axis=0)[None]

def _axes(x_data):
    return np.arange(len(x_data.shape) - 1) + 1

def _f_response(x_data, dt, gamma, ksq, a1=3., a2=0.):
    FX = lambda: fftn(x_data, _axes(x_data))
    FX3 = lambda: fftn(x_data ** 3, _axes(x_data))
    explicit = lambda: a1 - gamma * a2 * ksq
    implicit = lambda: (1 - gamma * ksq) - explicit
    dt_ = lambda: dt * ksq
    return (FX() * (1 + dt_() * explicit()) - dt_() * FX3()) / (1 - dt_() * implicit())

@curry
def solve_cahn_hilliard(x_data, dx=0.25, dt=0.001, gamma=4.):
    return ifftn(_f_response(_check(x_data),
                             dt,
                             gamma,
                             _calc_ksq(x_data, dx)),
                 _axes(x_data)).real

def _check(x_data):
    if not np.all(np.array(x_data.shape[1:]) == x_data.shape[1]):
        raise RuntimeError("X must represent a square domain")
    return x_data
