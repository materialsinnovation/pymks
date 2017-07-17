from .abstract import _AbstractMicrostructureBasis
from .fftmodule import rfftn, irfftn


class _RealFFTBasis(_AbstractMicrostructureBasis):
    """This class is used to make the bases that create real valued
    microstructure functions use the real rFFT/irFFT algorithms and selects
    the appropriate fft module depending on whether or not pyfftw is installed.
    """

    def _fftn(self, X):
        """Real rFFT algorithm

        Args:
            X: NDarray (n_samples, N_x, ...)

        Returns:
            Fourier transform of X
        """
        return rfftn(X, axes=self._axes, threads=self._n_jobs)

    def _ifftn(self, X):
        """Real irFFT algorithm

        Args:
            X: NDarray (n_samples, N_x, ...)

        Returns:
            Inverse Fourier transform of X
        """
        return irfftn(X, axes_shape=self._axes_shape, axes=self._axes, threads=self._n_jobs).real
