import numpy as np
from .abstract import _AbstractMicrostructureBasis


class _Polynomial(_AbstractMicrostructureBasis):

    def _regression_fit(self, X_, y):
        """
        Method used to calibrate influence coefficients from in
        mks_regresison_model.
        """
        axes = np.arange(X_.ndim)[1:-1]
        FX = np.fft.fftn(X_, axes=axes)
        Fy = np.fft.fftn(y, axes=axes)
        Fkernel = np.zeros(FX.shape[1:], dtype=np.complex)
        s0 = (slice(None),)
        for ijk in np.ndindex(X_.shape[1:-1]):
            Fkernel[ijk + s0] = np.linalg.lstsq(FX[s0 + ijk + s0],
                                                Fy[s0 + ijk])[0]
        return Fkernel
