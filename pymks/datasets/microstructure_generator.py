from ..filter import Filter
from scipy.ndimage.fourier import fourier_gaussian
from .base_microstructure_generator import _BaseMicrostructureGenerator
import numpy as np


class MicrostructureGenerator(_BaseMicrostructureGenerator):
    """
    Generates n_samples number of a periodic random microstructures
    with domain size equal to size and with n_phases number of
    phases. The optional grain_size argument controls the size and
    shape of the grains.

    >>> n_samples, n_phases = 1, 2
    >>> size = (4, 4)
    >>> generator = MicrostructureGenerator(n_samples, size,
    ...                                      n_phases, seed=10)
    >>> X = generator.generate()
    >>> X_test = np.array([[[1, 0, 1, 1],
    ...                     [1, 0, 0, 1],
    ...                     [0, 0, 1, 1],
    ...                     [0, 1, 1, 1]]])
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
        _gaussian_size = np.around(self.grain_size).astype(int)
        gaussian = fourier_gaussian(np.ones(_gaussian_size),
                                    np.ones(len(self.size)))
        filter_ = Filter(self._fftn(gaussian[None, ..., None]), self)
        filter_.resize(self.size)
        X_blur = filter_.convolve(X[..., None]).real
        return self._assign_phases(X_blur).astype(int)

    def _assign_phases(self, X_blur):
        """
        Takes in blurred array and assigns phase values.

        Args:
          X_blur: random field that has be blurred by a convolution.
        Returns:
          microstructure with assigned phases
        """
        if self.volume_fraction is None:
            epsilon = 1e-5
            X0, X1 = np.min(X_blur), np.max(X_blur)
            Xphases = float(self.n_phases) * ((X_blur - X0) / (X1 - X0) *
                                              (1. - epsilon) + epsilon)
            X_phases = np.floor(Xphases - epsilon)
        else:
            v_cum = np.cumsum(self.volume_fraction[:-1])
            X_sort = np.sort(X_blur.reshape((X_blur.shape[0], -1)), axis=1)
            seg_shape = (len(X_sort), len(v_cum))
            per_diff = (2 * np.random.random(seg_shape) -
                        1) * np.array(self.percent_variance)
            if -np.sum(per_diff) < self.percent_variance:
                per_diff -= np.sum(per_diff) / len(self.volume_fraction)
            seg_ind = np.floor((v_cum + per_diff) * X_sort.shape[1])
            seg_values = np.concatenate([x[list(i)][None]
                                         for i, x in zip(seg_ind, X_sort)])
            new_axes = [None for i in X_blur[0].shape]
            seg_slices = [slice(None)] + new_axes + [slice(None)]
            X_bool = X_blur[..., None] > seg_values[seg_slices]
            X_phases = np.sum(X_bool, axis=-1)
        return X_phases
