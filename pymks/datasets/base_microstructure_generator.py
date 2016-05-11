from ..bases.imag_ffts import _ImagFFTBasis
import numpy as np


class _BaseMicrostructureGenerator(_ImagFFTBasis):
    def __init__(self, n_samples=1, size=(21, 21),
                 n_phases=2, grain_size=None, seed=3, volume_fraction=None,
                 percent_variance=None):
        """
        Instantiate a MicrostructureGenerator.

        Args:
          n_samples (int): number of samples to be generated
          size (tuple): size of samples
          n_phases (int): number of phases in microstructures
          grain_size (tuple): size of the grain_size in the microstructure
          seed (int): seed for random number generator
          volume_fraction (tuple): the percent volume fraction for each phase.
          percent_variance (tuple): the percent variance for each value of
              volume_fraction. For example volume_fraction=(0.5, 0.5) and
              percent_variance=0.01 would be a 50 +/- 1 percent volume
              fractions for both values.

        Returns:
          n_samples number of a periodic random microstructure with size equal
          to size and with n_phases number of phases.
        """
        self._axes = np.arange(len(size)) + 1
        self._axes_shape = size
        self.n_samples = n_samples
        self.size = size
        self.n_phases = n_phases
        self.grain_size = grain_size
        if self.grain_size is None:
            self.grain_size = np.array(size) / 2
        np.random.seed(seed)
        self.volume_fraction = volume_fraction
        if self.volume_fraction is not None:
            if len(self.volume_fraction) != self.n_phases:
                raise RuntimeError(('n_phases and lenth of volume_fraction' +
                                   ' must be the same'))
            cum_frac = np.cumsum(volume_fraction)
            if not np.allclose(cum_frac[-1], 1):
                raise RuntimeError("volume fractions do not add up to 1")
            if percent_variance is None:
                percent_variance = 0.
            self.percent_variance = percent_variance
            min_frac = cum_frac[0] - percent_variance
            max_frac = cum_frac[-2] + percent_variance
            if max_frac > 1 or min_frac < 0:
                raise RuntimeError(('percent_variance cannot extend' +
                                    'volume_fraction values beyond 0 or 1'))
        super(_BaseMicrostructureGenerator, self).__init__()

    def generate(self):
        raise NotImplementedError
