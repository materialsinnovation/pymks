import numpy as np


def _module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


class Filter(object):

    """
    Wrapper class for convolution with a kernel and resizing of a kernel
    """

    def __init__(self, Fkernel):
        """
        Instantiate a Filter.

        Args:
          Fkernel: an array representing a convolution kernel
        """
        self._pyfftw = self._module_exists('pyfftw')
        self._fftmodule = self._load_fftmodule()
        self._axes = np.arange(len(Fkernel.shape) - 2) + 1
        self._Fkernel = Fkernel

    def _module_exists(self, module_name):
        try:
            __import__(module_name)
        except ImportError:
            return False
        else:
            return True

    def _load_fftmodule(self):
        if _module_exists('pyfftw'):
            import pyfftw.builders as fftmodule
        elif _module_exists('numpy.fft'):
            import numpy.fft as fftmodule
        else:
            raise RuntimeError('numpy or pyfftw cannot be imported')
        return fftmodule

    def _frequency_2_real(self):
        """
        Converts the kernel from frequency space to real space with
        the origin shifted to the center.

        Returns:
          an array in real space
        """
        return np.real_if_close(np.fft.fftshift(np.fft.ifftn(self._Fkernel,
                                                             axes=self._axes),
                                                axes=self._axes))

    def _real_2_frequency(self, kernel):
        """
        Converts a kernel from real space to frequency space.

        Args:
          kernel: an array representing a convolution kernel in real space

        Returns:
          an array in frequency space
        """
        return np.fft.fftn(np.fft.ifftshift(kernel, axes=self._axes),
                           axes=self._axes)

    def convolve(self, X):
        """
        Convolve X with a kernel in frequency space.

        Args:
          X: array to be convolved

        Returns:
          convolution of X with the kernel
        """
        if X.shape[1:] != self._Fkernel.shape[1:]:
            raise RuntimeError("Dimensions of X are incorrect.")
        FX = self._fftn(X, threads=3, avoid_copy=True)
        Fy = self._sum(FX * self._Fkernel)
        return self._ifftn(Fy).real

    def _sum(self, Fy):
        return np.sum(Fy, axis=-1)

    def resize(self, size):
        """
        Changes the size of the kernel to size.

        Args:
          size: tuple with the shape of the new kernel
        """
        if len(size) != len(self._Fkernel.shape[1:-1]):
            raise RuntimeError("length of resize shape is incorrect.")
        if not np.all(size >= self._Fkernel.shape[1:-1]):
            raise RuntimeError("resize shape is too small.")

        kernel = self._frequency_2_real()
        size = kernel.shape[:1] + size + kernel.shape[-1:]
        padsize = np.array(size) - np.array(kernel.shape)
        paddown = padsize // 2
        padup = padsize - paddown
        padarray = np.concatenate((padup[..., None],
                                   paddown[..., None]), axis=1)
        pads = tuple([tuple(p) for p in padarray])
        kernel_pad = np.pad(kernel, pads, 'constant', constant_values=0)
        Fkernel_pad = self._real_2_frequency(kernel_pad)
        self._Fkernel = Fkernel_pad

    def _fftn(self, X, s=None, threads=1, avoid_copy=False):
        if self._pyfftw:
            return self._fftmodule.fftn(X, axes=self._axes, s=s,
                                        threads=threads,
                                        planner_effort='FFTW_ESTIMATE',
                                        overwrite_input=True,
                                        avoid_copy=avoid_copy)()
        else:
            return self._fftmodule.fftn(X, axes=self._axes, s=s)

    def _ifftn(self, X, s=None, threads=1, avoid_copy=False):
        if self._pyfftw:
            return self._fftmodule.ifftn(X, axes=self._axes, s=s,
                                         threads=threads,
                                         planner_effort='FFTW_ESTIMATE',
                                         overwrite_input=True,
                                         avoid_copy=avoid_copy)()
        else:
            return self._fftmodule.ifftn(X, axes=self._axes, s=s)


class Correlation(Filter):

    """
    Computes the autocorrelation for a microstructure

    >>> n_states = 2
    >>> X = np.array([[[0, 1, 0],
    ...                [0, 1, 0],
    ...                [0, 1, 0]]])
    >>> from pymks.bases import DiscreteIndicatorBasis
    >>> basis = DiscreteIndicatorBasis(n_states=n_states)
    >>> X_ = basis.discretize(X)
    >>> filter_ = Correlation(X_)
    >>> X_auto = filter_.convolve(X_)
    >>> X_test = np.array([[[[3., 0.  ],
    ...                      [6., 3.],
    ...                      [3., 0.  ]],
    ...                     [[3., 0.  ],
    ...                      [6., 3.],
    ...                      [3., 0.  ]],
    ...                     [[3., 0.  ],
    ...                      [6., 3.],
    ...                      [3., 0.  ]]]])
    >>> assert(np.allclose(X_auto, X_test))

    Args:
        X_: The discretized microstructure function, an
            `(n_samples, n_x, ..., n_states)` shaped array
            where `n_samples` is the number of samples, `n_x` is thes
            patial discretization, and n_states is the number of local states.

    Returns:
        Autocorrelations for microstructure X_
    """

    def __init__(self, kernel, Fkernel_shape=None):
        self._axes = np.arange(len(kernel.shape) - 2) + 1
        self._pyfftw = self._module_exists('pyfftw')
        self._fftmodule = self._load_fftmodule()
        Fkernel = self._fftn(kernel, threads=3, s=Fkernel_shape)
        super(Correlation, self).__init__(np.conjugate(Fkernel))

    def convolve(self, X):
        """
        Convolve X with a kernel in frequency space.

        Args:
            X: array to be convolved

        Returns:
            correlation of X with the kernel
        """
        Fkernel_shape = np.array(self._Fkernel.shape)[self._axes]
        FX = self._fftn(X, s=Fkernel_shape, threads=3)
        Fy = self._sum(FX * self._Fkernel)
        correlation = self._ifftn(Fy).real
        return np.fft.fftshift(correlation, axes=self._axes)

    def _sum(self, Fy):
        return Fy
