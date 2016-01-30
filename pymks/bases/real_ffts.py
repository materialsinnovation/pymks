from .abstract import _AbstractMicrostructureBasis


class _RealFFTBasis(_AbstractMicrostructureBasis):
    def __init__(self, *args, **kwargs):
        super(_RealFFTBasis, self).__init__()

    def _fftn(self, X, s=None, threads=1, avoid_copy=False):
        if self._pyfftw:
            return self._fftmodule.rfftn(X, axes=self._axes, s=s,
                                         threads=threads,
                                         planner_effort='FFTW_ESTIMATE',
                                         overwrite_input=True,
                                         avoid_copy=avoid_copy)()
        else:
            return self._fftmodule.rfftn(X, axes=self._axes, s=s)

    def _ifftn(self, X, s=None, threads=1, avoid_copy=False):
        if self._pyfftw:
            return self._fftmodule.irfftn(X, axes=self._axes, s=s,
                                          threads=threads,
                                          planner_effort='FFTW_ESTIMATE',
                                          avoid_copy=avoid_copy)()
        else:
            return self._fftmodule.irfftn(X, axes=self._axes, s=s)

    def discretize(self, X):
        raise NotImplementedError
