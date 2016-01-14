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

    def _shape_check(self, X, y):
        if not len(y.shape) > 1:
            raise RuntimeError("The shape of y is incorrect.")
        if y.shape != X.shape:
            raise RuntimeError("X and y must be the same shape.")     

    def _output_shape(self,X):
        """
        Function to describe the expected output shape of a given
        microstructure X.
        """
        return X.shape

    def _select_slice(self, ijk, s0):
        """
        Helper method used to calibrate influence coefficients from in
        mks_localization_model to account for redundancies from linearly
        dependent local states.
        """
        return s0

    def _reshape_feature(self, X, size):
        """
        Helper function used to check the shape of the microstructure,
        and change to appropriate shape.

        Args:
            X: The microstructure, an `(n_samples, n_x, ...)` shaped array
                where `n_samples` is the number of samples and `n_x` is thes
                patial discretization.

        Returns:
            microstructure with shape (n_samples, size)
        """
        new_shape = (X.shape[0],) + size
        return X.reshape(new_shape)
