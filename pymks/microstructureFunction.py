import numpy as np
from sklearn.linear_model import LinearRegression


class MicrostructureFunction(LinearRegression):

    def __init__(self, n_states=None):
        '''Inits an `MicrostructureFunction`.

        Args:
            n_states: is the number of discretization states in the local
            state space.
        '''

        self.n_states = n_states

    def _axes(self, X):
        '''Generate argument for fftn.

        >>> X = np.zeros((5, 2, 2, 2))
        >>> print MicrostructureFunction()._axes(X)
        [1 2 3]

        Args:
            X: Array representing the microstructure.
        Returns:
            Array uses for axis argument in fftn.
        '''

        return np.arange(len(X.shape) - 1) + 1

    def _bin(self, X):
        '''Generate the microstructure function.

        Args:
            X: Array representing the Microstructure
        Returns:
            Microstructure function
        '''

        dim = len(X.shape) - 1
        if dim == 0:
            raise RuntimeError("the shape of X is incorrect")
        if issubclass(X.dtype.type, np.integer):
            Xbin = self._bin_int(X)
        else:
            Xbin = self._bin_float(X)
        return Xbin

    def _bin_float(self, X):
        '''
        >>> n_states = 10
        >>> np.random.seed(4)
        >>> X = np.random.random((2, 5, 3, 2))
        >>> #X = np.random.randint(n_states, size=(2, 5, 3, 2))
        >>> X_ = MicrostructureFunction(n_states)._bin(X)
        >>> H = np.linspace(0, 1, n_states)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)
        '''

        H = np.linspace(0, 1, self.n_states)
        return np.maximum(1 - (abs(X[..., None] - H)) / (H[1] - H[0]), 0)

    def _bin_int(self, X):
        '''Create microstruture function for integer valued microstrutures
        >>> MSf = MicrostructureFunction(3)
        >>> X = np.array([[1, 1, 0],
        ...               [1, 0 ,2],
        ...               [0, 1, 0]])

        >>> X_bin = np.array([[[0, 1, 0],
        ...                    [0, 1, 0],
        ...                    [1, 0, 0]],
        ...                   [[0, 1, 0],
        ...                    [1, 0, 0],
        ...                    [0, 0, 1]],
        ...                   [[1, 0, 0],
        ...                    [0, 1, 0],
        ...                    [1, 0, 0]]])
        >>> assert(np.allclose(X_bin, MSf._bin_int(X)))

        Args:
          X: Interger valued microstructure
        Returns:
          Interger valued microstructure function
        '''

        if np.min(X) != 0:
            raise RuntimeError("Phases must be zero indexed.")
        n_states = np.max(X) + 1
        if self.n_states is None:
            self.n_states = n_states
        if n_states != self.n_states:
            raise RuntimeError("Nphase does not correspond with phases in X.")
        Xbin = np.zeros(X.shape + (n_states,), dtype=float)
        mask = tuple(np.indices(X.shape)) + (X,)
        Xbin[mask] = 1.
        return Xbin

    def _binfft(self, X):
        '''Bin the microstructure and take the Fourier transform.

        >>> n_states = 10
        >>> np.random.seed(3)
        >>> X = np.random.random((2, 5, 3))
        >>> FX_ = MicrostructureFunction(n_states)._binfft(X)
        >>> X_ = np.fft.ifftn(FX_, axes=(1, 2))
        >>> H = np.linspace(0, 1, n_states)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        Args:
            X: Array representing the microstructure
        Returns:
            Microstructure function in frequency space
        '''

        Xbin = self._bin(X)
        return np.fft.fftn(Xbin, axes=self._axes(X))

    def _Legendre(self, X, deg, domain):
        '''This method takes in a continuous microstructure and returns a
        microstructure function with a Legendre polynomials as the basis.

        >>> deg = 3
        >>> X = np.array([[0.25, 0.1],
        ...               [0.5, 0.25]])
        >>> X_Legendre = np.array([[[-0.3125, -0.75, 0.5],
        ...                         [ 1.15,   -1.2, 0.5]],
        ...                        [[-1.25,      0, 0.5],
        ...                         [-0.3125, -0.75, 0.5]]])
        >>> MSf = MicrostructureFunction(n_states = deg)
        >>> assert(np.allclose(MSf._Legendre(X, 3, [0., 0.5]), X_Legendre))

        Args:
          X: Microstructure
          deg: The deg of Legendre polynomials used in the microstructure
               function.
          domain: The domain for the microstructure.
        Returns:
          Legendre polynomials up to deg evaluated at value of X for each cell
        '''

        leg = np.polynomial.legendre
        X_scaled = 2. * X - domain[0] - domain[1] / (domain[1] - domain[0])
        norm = (2. * np.arange(deg) + 1) / 2.
        X_Legendre = np.flipud(leg.legval(X_scaled, np.eye(deg) * norm))
        return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))

    def _Legendre_fft(self, X, deg, domain):
        '''Use Legendre polynomials basis for the microstructure function and
         take the Fourier transform.

        >>> n_states = 2
        >>> np.random.seed(3)
        >>> X = np.random.random((1, 3, 3))
        >>> MSf = MicrostructureFunction(n_states)
        >>> FXtest = MSf._Legendre_fft(X, n_states, [0, 1])
        >>> FX = np.array([[[[ 3.70264051+0.j,         -5.29735949+0.j    ],
        ...              [-1.00887157-1.48005289j, -1.00887157-1.48005289j],
        ...              [-1.00887157+1.48005289j, -1.00887157+1.48005289j]],
        ...             [[ 0.62300683-4.97732233j,  0.62300683-4.97732233j],
        ...              [ 1.09318216+0.10131035j,  1.09318216+0.10131035j],
        ...              [ 0.37713401+1.87334545j,  0.37713401+1.87334545j]],
        ...             [[ 0.62300683+4.97732233j,  0.62300683+4.97732233j],
        ...              [ 0.37713401-1.87334545j,  0.37713401-1.87334545j],
        ...              [ 1.09318216-0.10131035j,  1.09318216-0.10131035j]]]])
        >>> assert np.allclose(FX, FXtest)

        Args:
            X: Array representing the microstructure
        Returns:
            Microstructure function in frequency space
        '''

        X_Legendre = self._Legendre(X, deg, domain)
        return np.fft.fftn(X_Legendre, axes=self._axes(X_Legendre)[:-1])
