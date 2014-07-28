import numpy as np


class SpatialStatistics(object):

    def __init__(self, basis):
        '''
        Instantiate an SpatialStatistics.

        Args:
          basis: an instance of a bases class.
        '''
        self.basis = basis

    def Autocorrelation(self, X):
        '''
        >>> X = np.array([[[0, 1, 0],
        ...                [0, 1, 0],
        ... 			   [0, 1, 0]]])
        >>> from pymks.basis import DiscreteIndicatorBasis
        >>> basis = DiscreteIndicatorBasis()
        >>> X_auto = np.array([[]])
        '''
        X_ = self.basis.discretize(X)
        return self._fftconvolve(X_, X_)

    def _fftconvolve(self, X1_, X2_):
        '''
        Computes FFT along local state space dimension.

        '''
        axes = np.arange(len(X1_.shape) - 1) + 1
        FX1 = np.fft.fftn(X1_, axes=axes)
        FX2 = np.fft.fftn(X2_, axes=axes)
        norm = np.prod(np.array(X1_[0].shape))
        return np.fft.ifftn(np.conjugate(FX1) * FX2, axes=axes) / norm
