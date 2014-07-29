import numpy as np


class SpatialStatisticsModel(object):
    """
    The SpatialStatisticsModel takes in a microstructure and returns its two
    point statistics. Current the funciton only work for interger valued
    microstructures and the DiscreteIndicatorBasis.
    """

    def __init__(self, basis):
        '''
        Instantiate an SpatialStatistics.

        Args:
          basis: an instance of a bases class.
        '''
        self.basis = basis

    def get_autocorrelations(self, X):
        '''
        Computes the autocorrelation for a microstructure

        >>> n_states = 2
        >>> X = np.array([[[0, 1, 0],
        ...                [0, 1, 0],
        ... 			   [0, 1, 0]]])
        >>> from pymks.bases import DiscreteIndicatorBasis
        >>> basis = DiscreteIndicatorBasis(n_states=n_states)
        >>> from pymks import SpatialStatisticsModel
        >>> model = SpatialStatisticsModel(basis=basis)
        >>> X_auto = model.get_autocorrelations(X)
        >>> X_test = np.array([[[[1/3., 0.  ],
        ...                      [2/3., 1/3.],
        ...                      [1/3., 0.  ]],
        ...                     [[1/3., 0.  ],
        ...                      [2/3., 1/3.],
        ...                      [1/3., 0.  ]],
        ...                     [[1/3., 0.  ],
        ...                      [2/3., 1/3.],
        ...                      [1/3., 0.  ]]]])
        >>> assert(np.allclose(X_auto, X_test))

        Ags:
          X: microstructure
        Returns:
          Autocorrelations for microstructure X
        '''
        X_ = self.basis.discretize(X)
        axes = np.arange(len(X_.shape) - 2) + 1
        X_auto = self._fftconvolve(X_, X_, axes)
        norm = np.prod(np.array(X_auto[0, ..., 0].shape))
        return np.fft.fftshift(X_auto, axes=axes) / norm

    def get_crosscorrelations(self, X):
        '''
        Computes the crosscorrelations for a microstructure.

        >>> n_states = 2
        >>> X = np.array([[[0, 1, 0],
        ...                [0, 1, 0],
        ...                [0, 1, 0]]])
        >>> from pymks.bases import DiscreteIndicatorBasis
        >>> basis = DiscreteIndicatorBasis(n_states=n_states)
        >>> from pymks import SpatialStatisticsModel
        >>> model = SpatialStatisticsModel(basis=basis)
        >>> X_cross = model.get_crosscorrelations(X)
        >>> X_test = np.array([[[1/3.], [0.], [1/3.]],
        ...                    [[1/3.], [0.], [1/3.]],
        ...                    [[1/3.], [0.], [1/3.]]])
        >>> assert(np.allclose(X_cross, X_test))

        Args:
          X: microstructure
        Returns:
          Crosscorelations for microstructure X.
        '''
        X_ = self.basis.discretize(X)
        axes = np.arange(len(X_.shape) - 2) + 1
        X_shape = np.array(X_.shape)
        shape = tuple(X_shape[:-1]) + ((X_shape[-1] * (X_shape[-1] - 1) / 2),)
        X_cross = np.ones(shape)
        X_shift = X_.copy()
        index = self.basis.n_states - 1
        i_0, i_1 = 0, index
        for ii in range(index):
            X_shift = np.roll(X_shift, 1, axis=len(X_shape) - 1)
            X_cross[..., i_0:i_1] = self._fftconvolve(X_, X_shift,
                                                      axes=axes)[...,
                                                                 0:i_1 - i_0]
            i_0, i_1 = i_1, i_1 + index - (ii + 1)
        norm = np.prod(np.array(X_[0, ..., 0].shape))
        return np.fft.fftshift(X_cross, axes=axes) / norm

    def _fftconvolve(self, X1_, X2_, axes):
        '''
        Computes FFT along local state space dimension.

        '''
        FX1 = np.fft.fftn(X1_, axes=axes)
        FX2 = np.fft.fftn(X2_, axes=axes)
        X_result = np.fft.ifftn(np.conjugate(FX1) * FX2, axes=axes)
        return np.real_if_close(X_result, tol=1e6)
