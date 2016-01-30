from .abstract import _AbstractMicrostructureBasis


class _ImagFFTBasis(_AbstractMicrostructureBasis):
    def __init__(self, *args, **kwargs):
        super(_ImagFFTBasis, self).__init__(*args, **kwargs)

    def _fftn(self, X, s=None, threads=1, avoid_copy=False):
        if self._pyfftw:
            return self._fftmodule.fftn(X, axes=self._axes, s=s,
                                        threads=threads,
                                        planner_effort='FFTW_ESTIMATE',
                                        overwrite_input=True,
                                        avoid_copy=avoid_copy)()
        else:
            return self._fftmodule.fftn(X, axes=self._axes, s=s)

    def _ifftn(self, X, s=None, threads=1, avoid_copy=False):
        if self._pyfftw:
            return self._fftmodule.ifftn(X, axes=self._axes, s=s,
                                         threads=threads,
                                         planner_effort='FFTW_ESTIMATE',
                                         overwrite_input=True,
                                         avoid_copy=avoid_copy)()
        else:
            return self._fftmodule.ifftn(X, axes=self._axes, s=s)

    def discretize(self, X):
        raise NotImplementedError
