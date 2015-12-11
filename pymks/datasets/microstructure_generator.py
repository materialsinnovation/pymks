import numpy as np
from ..filter import Filter
from scipy.ndimage.fourier import fourier_gaussian
from .base_microstructure_generator import BaseMicrostructureGenerator
from ..filter import _import_pyfftw
_import_pyfftw()


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
        epsilon = 1e-5
        X0, X1 = np.min(X_blur), np.max(X_blur)
        Xphases = float(self.n_phases) * (X_blur - X0) / (X1 - X0) * \
                                         (1. - epsilon) + epsilon
        return np.floor(Xphases)
