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
        self._pyfftw = self._module_exists('pyfftw')
        self._fftmodule = self._load_fftmodule()

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

    def _load_fftmodule(self):
        if self._module_exists('pyfftw'):
            import pyfftw.builders as fftmodule
        elif self._module_exists('numpy.fft'):
            import numpy.fft as fftmodule
        else:
            raise RuntimeError('numpy or pyfftw cannot be imported')
        return fftmodule

    def discretize(self, X):
        raise NotImplementedError
