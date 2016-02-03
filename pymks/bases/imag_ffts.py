from .abstract import _AbstractMicrostructureBasis
import numpy as np


class _ImagFFTBasis(_AbstractMicrostructureBasis):
    def __init__(self, *args, **kwargs):
        super(_ImagFFTBasis, self).__init__(*args, **kwargs)

    def _fftn(self, X, n_jobs=1, avoid_copy=True):
        if self._pyfftw:
            return self._fftmodule.fftn(np.ascontiguousarray(X),
                                        axes=self._axes, threads=n_jobs,
                                        planner_effort='FFTW_ESTIMATE',
                                        overwrite_input=True,
                                        avoid_copy=avoid_copy)()
        else:
            return self._fftmodule.fftn(X, axes=self._axes)

    def _ifftn(self, X, s=None, n_jobs=1, avoid_copy=True):
        if self._pyfftw:
            return self._fftmodule.ifftn(np.ascontiguousarray(X),
                                         axes=self._axes, threads=n_jobs,
                                         planner_effort='FFTW_ESTIMATE',
                                         overwrite_input=True,
                                         avoid_copy=avoid_copy)().real
        else:
            return self._fftmodule.ifftn(X, axes=self._axes)

    def discretize(self, X):
        raise NotImplementedError
