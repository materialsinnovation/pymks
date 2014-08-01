import numpy as np
from pymks.filter import Filter
from scipy.ndimage.fourier import fourier_gaussian

class MicrostructureGenerator(object):
    """
    Generates n_samples number of a periodic random microstructures
    with domain size equal to size and with n_phases number of
    phases. The optional grain_size argument controls the size and
    shape of the grains.

    >>> n_samples, n_phases = 1, 2
    >>> size = (3, 3)
    >>> generator = MicrostructureGenerator(n_samples, size, n_phases, seed=10)
    >>> X = generator.generate()
    >>> X_test = np.array([[[1, 0, 1],
    ...                     [1, 1, 0],
    ...                     [0, 1, 0]]])
    >>> assert(np.allclose(X, X_test))

    """

    def __init__(self, n_samples, size, n_phases, grain_size=None, seed=3):
        """
        Instantiate a MicrostructureGenerator.
        
        Args:
          n_samples: number of samples to be generated
          size: size of samples
          n_phases: number of phases in microstructures
          grain_size: size of the grain_size in the microstructure
          seed: seed for random number generator
        
        Returns:
          n_samples number of a periodic random microstructure with size equal to
          size and with n_phases number of phases.
        """
        self.n_samples = n_samples
        self.size = size
        self.n_phases = n_phases
        self.grain_size = grain_size
        if self.grain_size is None:
            self.grain_size = np.array(size) / 2
        np.random.seed(seed)

    def generate(self):
        """
        Generates a microstructure of dimensions of self.size and grains
        with dimensions self.grain_size.

        Returns:
          periodic microstructure
        """
        if len(self.size) != len(self.grain_size):
            raise RuntimeError("Dimensions of size and grain_size are not equal.")
        X = np.random.random((self.n_samples,) + self.size)
        gaussian = fourier_gaussian(np.ones(self.grain_size), np.ones(len(self.size)))
        filter_ = Filter(np.fft.fftn(gaussian)[None,...,None])
        filter_.resize(self.size)
        X_blur = filter_.convolve(X[...,None])
        return self._assign_phases(X_blur).astype(int)

    def _assign_phases(self, X_blur):
        '''
        Takes in blurred array and assigns phase values.

        Args:
          X_blur: random field that has be blurred by a convolution.  
        Returns:
          microstructure with assigned phases
        '''
        epsilon = 1e-5
        X0, X1 = np.min(X_blur), np.max(X_blur)
        Xphases =  self.n_phases * (X_blur - X0) / (X1 - X0) * (1 - epsilon) + epsilon
        return np.floor(Xphases)

