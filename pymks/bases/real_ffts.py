from .abstract import _AbstractMicrostructureBasis
try:
    import pyfftw.builders as fftmodule
except:
    import numpy.fft as fftmodule
import numpy as np


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
        if self._pyfftw:
            return fftmodule.rfftn(np.ascontiguousarray(X),
                                   axes=self._axes, threads=self._n_jobs,
                                   planner_effort='FFTW_ESTIMATE',
                                   overwrite_input=True, avoid_copy=True)()
        else:
            return fftmodule.rfftn(X, axes=self._axes)

    def _ifftn(self, X):
        """Real irFFT algorithm

        Args:
            X: NDarray (n_samples, N_x, ...)

        Returns:
            Inverse Fourier transform of X
        """
        if self._pyfftw:
            return fftmodule.irfftn(np.ascontiguousarray(X),
                                    s=self._axes_shape, axes=self._axes,
                                    threads=self._n_jobs,
                                    planner_effort='FFTW_ESTIMATE',
                                    avoid_copy=True)().real
        else:
            return fftmodule.irfftn(X, axes=self._axes,
                                    s=self._axes_shape).real
