import numpy as np


class SpatialStatisticsModel(object):
    """
    The SpatialStatisticsModel takes in a microstructure and returns its two
    point statistics. Current the funciton only work for interger valued
    microstructures.
    """

    def __init__(self, basis):
        '''
        Instantiate an SpatialStatistics.

        Args:
          basis: an instance of a bases class.
        '''
        self.basis = basis

    def get_autocorrelation(self, X):
        '''
        >>> n_states = 2
        >>> X = np.array([[[0, 1, 0],
        ...                [0, 1, 0],
        ... 			   [0, 1, 0]]])
        >>> from pymks.bases import DiscreteIndicatorBasis
        >>> basis = DiscreteIndicatorBasis(n_states=n_states)
        >>> from pymks import SpatialStatisticsModel
        >>> SpatStats = SpatialStatisticsModel(basis=basis)
        >>> X_auto = SpatStats.get_autocorrelation(X)
        >>> X_test = np.array([[[[  1.66666667e-01,  -3.28954970e-17],
        ...                      [  1.66666667e-01,  -3.28954970e-17],
        ...                      [  3.33333333e-01,   1.66666667e-01]],
        ...                     [[  1.66666667e-01,  -3.28954970e-17],
        ...                      [  1.66666667e-01,  -3.28954970e-17],
        ...                      [  3.33333333e-01,   1.66666667e-01]],
        ...                     [[  1.66666667e-01,  -3.28954970e-17],
        ...                      [  1.66666667e-01,  -3.28954970e-17],
        ...                      [  3.33333333e-01,   1.66666667e-01]]]])
        >>> assert(np.allclose(X_auto, X_test))
        '''
        X_ = self.basis.discretize(X)
        axes = np.arange(len(X_.shape) - 2) + 1
        X_auto = np.real_if_close(self._fftconvolve(X_, X_, axes))
        return np.fft.ifftshift(X_auto, axes=axes)

    def _fftconvolve(self, X1_, X2_, axes):
        '''
        Computes FFT along local state space dimension.

        '''
        FX1 = np.fft.fftn(X1_, axes=axes)
        FX2 = np.fft.fftn(X2_, axes=axes)
        norm = np.prod(np.array(X1_[0].shape))
        return np.fft.ifftn(np.conjugate(FX1) * FX2, axes=axes) / norm
