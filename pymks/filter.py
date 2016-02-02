import numpy as np


class Filter(object):

    """
    Wrapper class for convolution with a kernel and resizing of a kernel
    """

    def __init__(self, Fkernel, basis, Fkernel_shape=None):
        """
        Instantiate a Filter.

        Args:
          Fkernel: an array representing a convolution kernel
        """
        self.basis = basis
        self._Fkernel = Fkernel
        self._kernel_shape = Fkernel_shape

    def _frequency_2_real(self):
        """
        Converts the kernel from frequency space to real space with
        the origin shifted to the center.

        Returns:
          an array in real space
        """
        return np.fft.fftshift(self.basis._ifftn(self._Fkernel,
                               s=self._kernel_shape[1:]),
                               axes=self.basis._axes)

    def _real_2_frequency(self, kernel):
        """
        Converts a kernel from real space to frequency space.

        Args:
          kernel: an array representing a convolution kernel in real space

        Returns:
          an array in frequency space
        """
        return self.basis._fftn(np.fft.ifftshift(kernel,
                                axes=self.basis._axes))

    def convolve(self, X):
        """
        Convolve X with a kernel in frequency space.

        Args:
          X: array to be convolved

        Returns:
          convolution of X with the kernel
        """
        FX = self.basis._fftn(X, threads=3, avoid_copy=True)
        if FX.shape[1:] != self._Fkernel.shape[1:]:
            raise RuntimeError("Dimensions of X are incorrect.")
        Fy = self._sum(FX * self._Fkernel)
        return self.basis._ifftn(Fy, s=X.shape[1:-1])

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
        kernel_pad = self._zero_pad(kernel, size)
        self._Fkernel = self._real_2_frequency(kernel_pad)
        self._kernel_shape = kernel_pad.shape[:-1]

    def _zero_pad(self, kernel, size):
        """
        Zero pads a real space array with zeros and does a Fourier transform

        Args:
            kernel: real space array

        Returns:
            Fourier transform of a zero padded kernel

        """
        if len(size) != kernel.ndim:
            size = kernel.shape[:1] + tuple(size) + kernel.shape[-1:]
        padsize = np.array(size) - np.array(kernel.shape)
        paddown = padsize // 2
        padup = padsize - paddown
        padarray = np.concatenate((padup[..., None],
                                   paddown[..., None]), axis=1)
        pads = tuple([tuple(p) for p in padarray])
        kernel_pad = np.pad(kernel, pads, 'constant', constant_values=0)
        return kernel_pad


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
    >>> filter_ = Correlation(X_, basis)
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

    def __init__(self, kernel, basis, Fkernel_shape=None):
        self.basis = basis
        if Fkernel_shape is not None:
            if Fkernel_shape > kernel.shape[1:-1]:
                kernel = self._zero_pad(kernel, Fkernel_shape)
        Fkernel = self.basis._fftn(kernel, threads=3)
        self._Fkernel_shape = Fkernel_shape
        super(Correlation, self).__init__(np.conjugate(Fkernel), basis,
                                          Fkernel_shape)

    def convolve(self, X):
        """
        Convolve X with a kernel in frequency space.

        Args:
            X: array to be convolved

        Returns:
            correlation of X with the kernel
        """
        if X.shape[1:-1] < self._Fkernel_shape:
            X = self._zero_pad(X, self._Fkernel_shape)
        FX = self.basis._fftn(X, threads=3)
        Fy = self._sum(FX * self._Fkernel)
        correlation = self.basis._ifftn(Fy, s=X.shape[1:-1])
        return np.fft.fftshift(correlation, axes=self.basis._axes)

    def _sum(self, Fy):
        return Fy
