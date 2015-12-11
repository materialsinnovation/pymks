import numpy as np


class _AbstractMicrostructureBasis(object):

    def __init__(self, n_states=2, domain=None):
        """
        Instantiate a `Basis`

        Args:
            n_states (int): The number of local states
            domain (list, optional): indicate the range of expected values for
                the microstructure, default is [0, n_states - 1].
        """
        self.n_states = n_states
        if domain is None:
            domain = [0, n_states - 1]
        self.domain = domain

    def check(self, X):
        if (np.min(X) < self.domain[0]) or (np.max(X) > self.domain[1]):
            raise RuntimeError("X must be within the specified domain")

    def discretize(self, X):
        raise NotImplementedError
