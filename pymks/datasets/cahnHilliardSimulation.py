import numpy as np

class CahnHilliardSimulation(object):
    r"""
    Solve the `Cahn-Hilliard equation`_ for multiple samples in
    arbitrary dimensions. The concentration varies from -1 to 1. The
    equation is given by

    .. math::

       \dot{\phi} = \nabla^2 \left( \phi^3 - \phi \right) - \nabla^4 \phi
       
    In 1D.
    
    >>> N = 100
    >>> phi = 0.01 * (2 * np.random.random((2, N)) - 1)
    >>> ch = CahnHilliardSimulation(width=2.)
    >>> for i in range(10000):
    ...     phi[:] = ch.get_response(phi)
    >>> assert (max(phi.flat) > 0.95) and (min(phi.flat) < -0.95)

    In 2D.

    >>> phi = 0.01 * (2 * np.random.random((2, N, N)) - 1)
    >>> ch = CahnHilliardSimulation(width=2.)
    >>> for i in range(100):
    ...     phi[:] = ch.get_response(phi)
    >>> assert (max(phi.flat) > 0.001) and (min(phi.flat) < -0.001)    

    In 3D.

    >>> phi = 0.01 * (2 * np.random.random((2, N, N, N)) - 1)
    >>> ch = CahnHilliardSimulation(width=2.)
    >>> for i in range(10):
    ...     phi[:] = ch.get_response(phi)
    >>> assert (max(phi.flat) > 0.0005) and (min(phi.flat) < -0.0005)    

    _`Cahn-Hilliard equation`: https://en.wikipedia.org/wiki/Cahn-Hilliard_equation
    
    """

    def __init__(self, dx=0.5, width=1., dt=1.):
        r"""
        Args:
          dx: grid spacing
          dt: time step size
          width: interface width
          
        """
        self.dx = dx
        self.dt = dt
        self.width = width

    def get_response(self, X):
        r"""
        Return the response field

        Args:
          X: Array representing the concentration field between -1 and
             1 with shape (Nsample, N, N)

        Returns:
          Array representing the microstructure at one time step ahead
          of 'X'

        """
        N = X.shape[1]
        if not np.all(np.array(X.shape[1:]) == N):
            raise RuntimeError, 'X must represent a square domain'

        L = self.dx * N
        k = np.arange(N)
        k[N / 2:] = (k - N / 2)[:N / 2]
        k = k * 2 * np.pi / L

        i_ = np.indices(X.shape[1:])
        ksq = np.sum(k[i_]**2, axis=0)[None]

        axes = np.arange(len(X.shape) - 1) + 1
        FX = np.fft.fftn(X, axes=axes)
        FX3 = np.fft.fftn(X**3, axes=axes)
        
        a1 = 3.
        a2 = 0.
        gamma = self.width**2
        explicit = ksq * (a1 - gamma * a2 * ksq)
        implicit = ksq * ((1 - a1) - gamma * (1 - a2) * ksq)
        dt = self.dt

        Fy = (FX * (1 + dt * explicit) - ksq * dt * FX3) / (1 - dt * implicit)
        return np.fft.ifftn(Fy, axes=axes).real
