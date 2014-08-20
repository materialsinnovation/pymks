import numpy as np


class _AbstractMicrostructureBasis(object):
    def __init__(self, n_states=1, domain=[0, 1]):
        """
        Instantiate a `Basis`

        Args:
          n_states: The number of local states
          domain: indicate the range of expected values for the microstructure.
        """
        self.n_states = n_states
        self.domain = domain
        
    def check(self, X):
        if (np.min(X) < self.domain[0]) or (np.max(X) > self.domain[1]):
            raise RuntimeError("X must be within the specified range")

    def discretize(self, X):
        raise NotImplementedError

