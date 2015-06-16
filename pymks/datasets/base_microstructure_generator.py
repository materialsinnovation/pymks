import numpy as np


class BaseMicrostructureGenerator(object):
    def __init__(self, n_samples=1, size=(21, 21),
                 n_phases=2, grain_size=None, seed=3):
        """
        Instantiate a MicrostructureGenerator.

        Args:
          n_samples: number of samples to be generated
          size: size of samples
          n_phases: number of phases in microstructures
          grain_size: size of the grain_size in the microstructure
          seed: seed for random number generator

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

    def generate(self):
        raise NotImplementedError
