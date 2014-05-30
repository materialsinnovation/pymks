from .mksRegressionModel import MKSRegressionModel
import numexpr as ne
import numpy as np


class FastMKSRegressionModel(MKSRegressionModel):

    r"""
    This class is an optimized version of MKSRegressionModel class.
    """

    def __init__(self, Nbin=10, threads=1):
        r"""
        Create a `FastMKSRegressionModel`.

        Args:
            Nbin: is the number of discretization bins for the
                "microstructure function".
            threads: the number of threads to use for multi-threading.

        """
        super(FastMKSRegressionModel, self).__init__(Nbin=Nbin)
        self.threads = threads
        ne.set_num_threads(threads)

    def _bin(self, X):
        """
        Bin the microstructure.

        >>> Nbin = 10
        >>> np.random.seed(4)
        >>> X = np.random.random((2, 5, 3, 2))
        >>> X_ = FastMKSRegressionModel(Nbin)._bin(X)
        >>> H = np.linspace(0, 1, Nbin)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        Args:
            X: Array representing the microstructure
        Returns:
            Array representing the microstructure function
        """
        H = np.linspace(0, 1, self.Nbin)
        dh = H[1] - H[0]
        Xtmp = X[..., None]
        tmp = ne.evaluate("1. - abs(Xtmp - H) / dh")
        return ne.evaluate("(tmp > 0) * tmp")

    def _binfft(self, X):
        r"""
        Bin the microstructure and take the Fourier transform.

        >>> Nbin = 10
        >>> np.random.seed(3)
        >>> X = np.random.random((2, 5, 3))
        >>> FX_ = FastMKSRegressionModel(Nbin)._binfft(X)
        >>> X_ = np.fft.ifftn(FX_, axes=(1, 2))
        >>> H = np.linspace(0, 1, Nbin)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        Args:
            X: Array representing the microstructure.
        Returns:
            Array representing the microstructure function in
            in frequency space.

        """
        Xbin = self._bin(X)
        return self._fftn(Xbin, axes=self._axes(X))

    def _fftplan(self, a, axes, direction='FFTW_FORWARD'):
        r"""
        Helper function used in _calcfft
        """
        import pyfftw
        input_array = pyfftw.n_byte_align_empty(a.shape, 16, 'complex128')
        output_array = pyfftw.n_byte_align_empty(a.shape, 16, 'complex128')
        return pyfftw.FFTW(
            input_array, output_array, threads=self.threads, axes=axes, direction=direction)

    def _calcfft(self, a, axes, direction='FFTW_FORWARD'):
        r"""
        Helper function that does the performs the fast fourier
        transform algorithm.

        Args:
            a: Array that will be transformed to frequency space.
            axes: Dimension of `a` that will be transformed
            direction: The direction of the transform
        Returns:
            An array that has the dimensions `axes` transformed to
            frequency space.

        """
        if hasattr(self, direction):
            input_array = getattr(self, direction).get_input_array()
            if input_array.shape != a.shape:
                setattr(self, direction, self._fftplan(a, axes, direction))
        else:
            setattr(self, direction, self._fftplan(a, axes, direction))

        input_array = getattr(self, direction).get_input_array()
        input_array[:] = a
        return getattr(self, direction)()

    def _fftn(self, a, axes):
        r"""
        Computes the FFT of `a` along dimensions `axes`.

        Args:
            a: Array to be transformed
            axes: Dimension where the transform will occur
        Returns:
            An array that has been transformed from real space
            to frequency space along dimensions `axes`.
        """
        return self._calcfft(a, axes, direction='FFTW_FORWARD')

    def _ifftn(self, a, axes):
        r"""
        Computes the iFFT of `a` along dimensions `axes`.

        Args:
            a: Array to be transformed
            axes: Dimensions where the transform will occur
        Returns:
            An array that has been transformed from frequency space
            to real space along dimensions `axes`.
        """
        return self._calcfft(a, axes, direction='FFTW_BACKWARD')

    def predict(self, X):
        r"""
        Calculates a response from the microstructure `X`.

        >>> X = np.linspace(0, 1, 4).reshape((1, 2, 2))
        >>> y = X.swapaxes(1, 2)
        >>> model = FastMKSRegressionModel(Nbin=2)
        >>> model.fit(X, y)
        >>> assert np.allclose(y, model.predict(X))

        Args:
            X: The microstructure function, an `(S, N, ...)` shaped
                array where `S` is the number of samples and `N`
                is the spatial discretization.

        Returns:
            The predicted response field the same shape as `X`

        """
        assert X.shape[1:] == self.Fcoeff.shape[:-1]
        FX = self._binfft(X)
        tmp = self.Fcoeff[None, ...]
        axis = len(tmp.shape) - 1
        tmp = ne.evaluate("FX * tmp")
        Fy = ne.evaluate("sum(tmp, axis={0})".format(axis))
        return self._ifftn(Fy, axes=self._axes(X)).real


