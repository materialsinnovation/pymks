import numpy as np


class BaseMicrostructureGenerator(object):
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
        self.n_samples = n_samples
        self.size = size
        self.n_phases = n_phases
        self.grain_size = grain_size
        if self.grain_size is None:
            self.grain_size = np.array(size) / 2
        np.random.seed(seed)
        self.volume_fraction = volume_fraction
        if self.volume_fraction is not None:
            if not np.allclose(np.array(np.cumsum(volume_fraction)[-1],
                               np.array([1]))):
                raise RuntimeError("volume fractions do not add up to 1")
            if percent_variance is None:
                percent_variance = np.zeros(len(volume_fraction))
            self.percent_variance = percent_variance

    def generate(self):
        raise NotImplementedError
