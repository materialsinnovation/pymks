from ..bases.imag_ffts import _ImagFFTBasis
import numpy as np


class CahnHilliardSimulation(_ImagFFTBasis):
    r"""
    Solve the `Cahn-Hilliard equation
    <https://en.wikipedia.org/wiki/Cahn-Hilliard_equation>`__ for
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
    >>> N = 100
    >>> phi = 0.01 * (2 * np.random.random((2, N)) - 1)
    >>> ch = CahnHilliardSimulation(gamma=4)
    >>> for i in range(10000):
    ...     ch.run(phi)
    ...     phi[:] = ch.response
    >>> assert (max(phi.flat) > 2e-3) and (min(phi.flat) < -2e-3)

    In 2D.

    >>> N = 101
    >>> phi = 0.01 * (2 * np.random.random((2, N, N)) - 1)
    >>> ch = CahnHilliardSimulation(gamma=4.)
    >>> for i in range(100):
    ...     ch.run(phi)
    ...     phi[:] = ch.response
    >>> assert (max(phi.flat) > 0.001) and (min(phi.flat) < -0.001)

    In 3D.

    >>> phi = 0.01 * (2 * np.random.random((2, N, N, N)) - 1)
    >>> ch = CahnHilliardSimulation(gamma=4.)
    >>> for i in range(10):
    ...     ch.run(phi)
    ...     phi[:] = ch.response

    >>> assert (max(phi.flat) > 0.0005) and (min(phi.flat) < -0.0005)

    """

    def __init__(self, dx=0.25, gamma=1., dt=0.001):
        r"""
        Instanitate a CahnHilliardSimulation

        Args:
            dx (float, optional): grid spacing
            dt (float, optional): time step size
            gamma (float, optional): paramater in CH equation

        """
        self.dx = dx
        self.dt = dt
        self.gamma = gamma
        super(CahnHilliardSimulation, self).__init__()

    def run(self, X):
        r"""
        Return the response field

        Args:
            X (ND array): Array representing the concentration field between -1
                and 1 with shape (n_samples, n_x, ...)

        """
        N = X.shape[1]
        if not np.all(np.array(X.shape[1:]) == N):
            raise RuntimeError("X must represent a square domain")

        L = self.dx * N
        k = np.arange(N)

        if N % 2 == 0:
            N1 = N / 2
            N2 = N1
        else:
            N1 = (N - 1) / 2
            N2 = N1 + 1

        k[N2:] = (k - N1)[:N1]
        k = k * 2 * np.pi / L

        i_ = np.indices(X.shape[1:])
        ksq = np.sum(k[i_] ** 2, axis=0)[None]

        self._axes = np.arange(len(X.shape) - 1) + 1
        FX = self._fftn(X)
        FX3 = self._fftn(X ** 3)

        a1 = 3.
        a2 = 0.
        explicit = ksq * (a1 - self.gamma * a2 * ksq)
        implicit = ksq * ((1 - a1) - self.gamma * (1 - a2) * ksq)
        dt = self.dt

        Fy = (FX * (1 + dt * explicit) - ksq * dt * FX3) / (1 - dt * implicit)
        self.response = self._ifftn(Fy).real
