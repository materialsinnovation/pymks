import numpy as np
from ..filter import Filter
from scipy.ndimage.fourier import fourier_gaussian
from .base_microstructure_generator import BaseMicrostructureGenerator
from ..filter import _import_pyfftw
_import_pyfftw()
import math
import datetime


class MicrostructureGenerator(BaseMicrostructureGenerator):
    """
    Generates n_samples number of a periodic random microstructures
    with domain size equal to size and with n_phases number of
    phases. The optional grain_size argument controls the size and
    shape of the grains.

    >>> n_samples, n_phases = 1, 2
    >>> size = (3, 3)
    >>> generator = MicrostructureGenerator(n_samples, size,
    ...                                      n_phases, seed=10)
    >>> X = generator.generate()
    >>> X_test = np.array([[[1, 0, 1],
    ...                     [1, 1, 0],
    ...                     [0, 1, 0]]])
    >>> assert(np.allclose(X, X_test))

    """

    def generate(self):
        """
        Generates a microstructure of dimensions of self.size and grains
        with dimensions self.grain_size.

        Returns:
          periodic microstructure
        """
        if len(self.size) != len(self.grain_size):
            raise RuntimeError("Dimensions of size and grain_size are"
                               " not equal.")

        X = np.random.random((self.n_samples,) + self.size)
        gaussian = fourier_gaussian(np.ones(self.grain_size),
                                    np.ones(len(self.size)))
        filter_ = Filter(np.fft.fftn(gaussian)[None, ..., None])
        filter_.resize(self.size)
        X_blur = filter_.convolve(X[..., None])
        return self._assign_phases(X_blur).astype(int)

    def _assign_phases(self, X_blur):
        """
        Takes in blurred array and assigns phase values.

        Args:
          X_blur: random field that has be blurred by a convolution.
        Returns:
          microstructure with assigned phases
        """
        '''
        Former version    
            if self.n_phases > 3:
            epsilon = 1e-5
            X0, X1 = np.min(X_blur), np.max(X_blur)
            Xphases = float(self.n_phases) * (X_blur - X0) / (X1 - X0) * \
                                         (1. - epsilon) + epsilon
        '''
        v_cum = np.cumsum(self.volume_fraction)
        if v_cum[-1] > 1:
            raise RuntimeError("Volume fractions do not add up to 1")
        X_reshape = X_blur.reshape((X_blur.shape[0], -1))
        X_sort = np.sort(X_reshape, axis=1)
        X_segs = np.zeros((X_reshape.shape))
        length = X_sort.shape[1]
        
        
        #v_boundaries = np.concatenate((v_cum[None], np.roll(v_cum, -1)[None]))
        #v_slices = np.mean(v_boundaries, axis=0)
        #v_slices[0] = 0
        #v_slices[-1] = 1
        #_slices = [slice(np.floor(i * length) , np.floor(j * length))
        #           for i, j in zip(v_slices[:-1], v_slices[1:])]
        
        #print _slices
        #print v_cum[:-1]
        seg_shape = (length, len(v_cum[1:]))
        per_diff = self.percent_variance * (2 * np.random.random((seg_shape)) - 1)
        seg_ind = np.floor((v_cum[:-1] + per_diff) * length)
        seg_values = np.concatenate([x[list(i)][None]
                                     for i, x in zip(seg_ind, X_sort)])
        X_bool = np.less(X_blur[..., None], seg_values[:, None, None, :])
        X_phases = np.sum(X_bool, axis=-1)
        #ind = int(math.floor(v_cum*length))
        #print X_blur.shape
        #print np.unique(X_phases)
            #for i in range(1,len(v_frac)):
            #t= datetime.datetime.now()
            #np.random.seed((t.microsecond))
            #v = v_cum[i]
            #
            #var = self.sigma*np.random.randn()
            #print "variance = " + str(var)
            ##index = ind + var
            #print "Index = " + str(index)
            #print X_sort.shape
            #seg = X_sort[:, index]
            #X_seg = X_reshape >= seg[:, None]
            #X_segs = X_segs+X_seg
            #Xphases = X_segs.reshape((X_blur.shape))


        return X_phases
