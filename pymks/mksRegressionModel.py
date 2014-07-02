import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
mse = metrics.mean_squared_error


class MKSRegressionModel(LinearRegression):

    r"""
    The `MKSRegressionModel` fits data using the Materials Knowledge
    System in Fourier Space. Currently, the model assumes that the
    microstructure (`X`) varies only between 0 and 1.

    The following demonstrates the viability of the
    `MKSRegressionModel` with a simple 1D filter.

    >>> Nbin = 2
    >>> Nspace = 81
    >>> Nsample = 400

    Define a filter function.

    >>> def filter(x):
    ...     return np.where(x < 10,
    ...                     np.exp(-abs(x)) * np.cos(x * np.pi),
    ...                     np.exp(-abs(x - 20)) * np.cos((x - 20) * np.pi))

    Use the filter function to construct some coefficients.

    >>> coeff = np.linspace(1, 0, Nbin)[None,:] * filter(np.linspace(0, 20, Nspace))[:,None]
    >>> Fcoeff = np.fft.fft(coeff, axis=0)

    Make some test samples.

    >>> np.random.seed(2)
    >>> X = np.random.random((Nsample, Nspace))

    Construct a response with the `Fcoeff`.

    >>> H = np.linspace(0, 1, Nbin)
    >>> X_ = np.maximum(1 - abs(X[:,:,None] - H) / (H[1] - H[0]), 0)
    >>> FX = np.fft.fft(X_, axis=1)
    >>> Fy = np.sum(Fcoeff[None] * FX, axis=-1)
    >>> y = np.fft.ifft(Fy, axis=1).real

    Use the `MKSRegressionModel` to reconstruct the coefficients

    >>> model = MKSRegressionModel(Nbin=Nbin)
    >>> #model.fit(X, y)

    Check the result

    >>> #assert np.allclose(np.fft.fftshift(coeff, axes=(0,)), model.coeff)

    Attributes:
        Nbin: Interger value for number of local states
        coef: Array of values that are the influence coefficients
        Fcoef: Frequency space representation of coef
    """

    def __init__(self, Nbin=10):
        r"""
        Inits an `MKSRegressionModel`.

        Args:
            Nbin: is the number of discretization bins in the local
            state space.
        """
        self.Nbin = Nbin

    def _axes(self, X):
        r"""

        Generate argument for fftn.

        >>> X = np.zeros((5, 2, 2, 2))
        >>> print MKSRegressionModel()._axes(X)
        [1 2 3]

        Args:
            X: Array representing the microstructure.
        Returns:
            Array uses for axis argument in fftn.

        """
        return np.arange(len(X.shape) - 1) + 1

    def _bin(self, X):
        """
        Generate the microstructure function.

        Args:
            X: Array representing the Microstructure
        Returns:
            Microstructure function
        """
        dim = len(X.shape) - 1
        if dim == 0:
            raise RuntimeError, "the shape of X is incorrect"
        if issubclass(X.dtype.type, np.integer):
            Xbin = self._bin_int(X)
        else:
            Xbin = self._bin_float(X)
        return Xbin

    def _bin_float(self, X):
        '''
        >>> Nbin = 10
        >>> np.random.seed(4)
        >>> X = np.random.random((2, 5, 3, 2))
        >>> #X = np.random.randint(Nbin, size=(2, 5, 3, 2))
        >>> X_ = MKSRegressionModel(Nbin)._bin(X)
        >>> H = np.linspace(0, 1, Nbin)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)
        
        '''
        H = np.linspace(0, 1, self.Nbin)
        return np.maximum(1 - (abs(X[..., None] - H)) / (H[1] - H[0]), 0)

    def _bin_int(self, X):
        '''
        >>> MKSmodel = MKSRegressionModel(Nbin=3)
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
        
        >>> assert(np.allclose(X_bin, MKSmodel._bin_int(X)))

        '''
        if np.min(X) != 0:
            raise RuntimeError, "Phases must be zero indexed."
        Nphase = np.max(X) + 1
        Xbin = np.zeros(X.shape + (Nphase,), dtype=float)
        mask = tuple(np.indices(X.shape)) + (X,)
        Xbin[mask] = 1.
        return Xbin
        

    def _binfft(self, X):
        r"""
        Bin the microstructure and take the Fourier transform.

        >>> Nbin = 10
        >>> np.random.seed(3)
        >>> X = np.random.random((2, 5, 3))
        >>> FX_ = MKSRegressionModel(Nbin)._binfft(X)
        >>> X_ = np.fft.ifftn(FX_, axes=(1, 2))
        >>> H = np.linspace(0, 1, Nbin)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        Args:
            X: Array representing the microstructure
        Returns:
            Microstructure function in frequency space

        """
        Xbin = self._bin(X)
        return np.fft.fftn(Xbin, axes=self._axes(X))

    def fit(self, X, y):
        r"""
        Fits the data by calculating a set of influence coefficients,
        `Fcoeff`.

        >>> X = np.linspace(0, 1, 4).reshape((1, 2, 2))
        >>> y = X.swapaxes(1, 2)
        >>> model = MKSRegressionModel(Nbin=2)
        >>> model.fit(X, y)
        >>> assert np.allclose(model.Fcoeff, [[[ 0.5,  0.5], [-2, 0]],
        ...                                   [[-0.5,  0  ], [-1, 0]]])


        Args:
            X: the microstructre function, an `(S, N, ...)` shaped array where
                `S` is the number of samples and `N` is the spatial
               discretization.
            y: The response field, same shape as `X`.
        """

        assert len(y.shape) > 1
        assert y.shape == X.shape
        FX = self._binfft(X)
        Fy = np.fft.fftn(y, axes=self._axes(X))
        shape = X.shape[1:]
        self.Fcoeff = np.zeros(shape + (self.Nbin,), dtype=np.complex)
        s0 = (slice(None),)
        for ijk in np.ndindex(shape):
            if np.all(np.array(ijk) == 0):
                s1 = s0
            else:
                s1 = (slice(-1),) 
            self.Fcoeff[ijk + s1] = np.linalg.lstsq(FX[s0 + ijk + s1],
                                                    Fy[s0 + ijk])[0]

    @property
    def coeff(self):
        axes = np.arange(len(self._axes(self.Fcoeff) - 1))
        return np.real_if_close(np.fft.fftshift(np.fft.ifftn(self.Fcoeff, axes=axes), axes=axes))

    def coeffToFcoeff(self, coeff):
        axes = np.arange(len(self._axes(self.Fcoeff) - 1))
        return np.fft.fftn(np.fft.ifftshift(coeff, axes=axes), axes=axes)

    def predict(self, X):
        r"""
        Calculate a new response from the microstructure function `X` with calibrated
        influence coefficients.

        >>> X = np.linspace(0, 1, 4).reshape((1, 2, 2))
        >>> y = X.swapaxes(1, 2)
        >>> model = MKSRegressionModel(Nbin=2)
        >>> model.fit(X, y)
        >>> assert np.allclose(y, model.predict(X))

        Args:
            X: The microstructre function, an `(S, N, ...)` shaped
                array where `S` is the number of samples and `N`
                is the spatial discretization.

        Returns:
            The predicted response field the same shape as `X`.

        """
        assert X.shape[1:] == self.Fcoeff.shape[:-1]
        FX = self._binfft(X)
        Fy = np.sum(FX * self.Fcoeff[None, ...], axis=-1)
        return np.fft.ifftn(Fy, axes=self._axes(X)).real

    def resize_coeff(self, shape):
        r"""
        Scale the size of the coefficients and pad with zeros.

        Let's first instantitate a model and fabricate some
        coefficients.

        >>> model = MKSRegressionModel()
        >>> coeff = np.arange(20).reshape((5, 4, 1))
        >>> coeff = np.concatenate((coeff , np.ones_like(coeff)), axis=2)
        >>> coeff = np.fft.ifftshift(coeff, axes=(0, 1))
        >>> model.Fcoeff = np.fft.fftn(coeff, axes=(0, 1))

        The coefficients can be reshaped by passing the new shape that
        coefficients should have.

        >>> model.resize_coeff((10, 7))
        >>> assert np.allclose(model.coeff[:,:,0],
        ...                    [[0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 1, 2, 3, 0],
        ...                     [0, 0, 4, 5, 6, 7, 0],
        ...                     [0, 0, 8, 9,10,11, 0],
        ...                     [0, 0,12,13,14,15, 0],
        ...                     [0, 0,16,17,18,19, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0]])

        Args:
            shape: The new shape of the influence coefficients.
        Returns:
            The resized influence coefficients to size 'shape'.

        """
        if len(shape) != (len(self.Fcoeff.shape) - 1):
            raise RuntimeError, 'length of resize shape is incorrect'
        if not np.all(shape >= self.Fcoeff.shape[:-1]):
             raise RuntimeError, 'resize shape is too small'

        coeff = self.coeff
        shape += coeff.shape[-1:]
        padsize = np.array(shape) - np.array(coeff.shape)
        paddown = padsize / 2
        padup = padsize - paddown
        padarray = np.concatenate((padup[...,None], paddown[...,None]), axis=1) 
        pads = tuple([tuple(p) for p in padarray])
        coeff_pad = np.pad(coeff, pads, 'constant', constant_values=0)
        Fcoeff_pad = self.coeffToFcoeff(coeff_pad)
        Fcoeff_pad[...,-1] = 0
        k0 = np.nonzero(self.Fcoeff[...,-1])
        Fcoeff_pad[...,-1][k0] = self.Fcoeff[...,-1][k0]
        
        self.Fcoeff = Fcoeff_pad

    def _test(self):
        r"""
        Test with a Cahn-Hilliard model.

        >>> from pymks import FiPyCHModel
        >>> Nsample = 100
        >>> Nspace = 21
        >>> dt = 1e-3
        >>> np.random.seed(0 )
        >>> X = np.array([np.random.random((Nspace, Nspace)) for i in range(Nsample)])
        >>> fipy_model = FiPyCHModel(dx=0.25, dy=0.25, dt=1e-3, epsilon=1., a=1.)
        >>> y = fipy_model.predict(X)
        >>> model = MKSRegressionModel(Nbin=10)
        >>> model.fit(X, y)
        >>> X_test = np.array([np.random.random((Nspace, Nspace)) for i in range(1)])
        >>> y_test = fipy_model.predict(X_test)
        >>> y_pred = model.predict(X_test)
        >>> assert mse(y_test, y_pred) < 0.03
        """
        pass
