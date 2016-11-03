from .abstract import _AbstractMicrostructureBasis
from .fftmodule import ifftn, fftn


class _ImagFFTBasis(_AbstractMicrostructureBasis):
    """This class is used to make the bases that create complex valued
    microstructure functions use the standard FFT/iFFT algorithms and selects
    the appropriate fft module depending on whether or not pyfftw is installed.
    """

    def _fftn(self, X):
        """Standard FFT algorithm

        Args:
            X: NDarray (n_samples, N_x, ...)

        Returns:
            Fourier transform of X
        """
        return fftn(X, axes=self._axes, threads=self._n_jobs)

    def _ifftn(self, X):
        """Standard iFFT algorithm

        Args:
            X: NDarray (n_samples, N_x, ...)

        Returns:
            Inverse Fourier transform of X
        """
        return ifftn(X, axes=self._axes, threads=self._n_jobs)
