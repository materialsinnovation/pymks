try:
    import pyfftw
except:
    pass
import numpy as np


class _AbstractMicrostructureBasis(object):

    def __init__(self, n_states=2, domain=None):
        """
        Instantiate a `Basis`

        Args:
            n_states (int, list): The number of local states, or an array of
                local states to be used.
            domain (list, optional): indicate the range of expected values for
                the microstructure, default is [0, n_states - 1].
        """
        self.n_states = n_states
        if isinstance(self.n_states, int):
            self.n_states = np.arange(n_states)
        if domain is None:
            domain = [0, max(self.n_states)]
        self.domain = domain
        self._pyfftw = self._module_exists('pyfftw')
        self._n_jobs = 1

    def check(self, X):
        if (np.min(X) < self.domain[0]) or (np.max(X) > self.domain[1]):
            raise RuntimeError("X must be within the specified domain")

    def _module_exists(self, module_name):
        try:
            __import__(module_name)
        except ImportError:
            return False
        else:
            return True

    def discretize(self, X):
        raise NotImplementedError

    def _check_shape(self, X_shape, y_shape):
        if not len(y_shape) > 1:
            raise RuntimeError("The shape of y is incorrect.")
        if y_shape != X_shape:
            raise RuntimeError("X and y must be the same shape.")

    def _pred_shape(self, X):
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
                where `n_samples` is the number of samples and `n_x` is the
                patial discretization.
            size: the new size of the array

        Returns:
            microstructure with shape (n_samples, size)
        """
        return X.reshape((X.shape[0],) + size)

    def _reshape_localization_data(self, y, size):
        """
        Helper function used to check the shape of the microstructure,
        and change to appropriate shape.

        Args:
            y: The localization fields, an `(n_samples, n_x, ...)` shaped array
                where `n_samples` is the number of samples and `n_x` is thes
                patial discretization.
            size: the new size of the array

        Returns:
            Localization fields with shape (n_samples, size)
        """
        return y.reshape((y.shape[0],) + size)

    def _select_axes(self, X):
        self._axes = np.arange(X.ndim - 1) + 1
        self._axes_shape = X[0].shape
