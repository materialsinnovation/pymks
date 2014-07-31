import numpy as np


class Filter(object):
    """
    Wrapper class for convolution with a kernel and resizing of a kernel
    """
    def __init__(self, Fkernel, axes=None):
        """
        Instantiate a Filter.

        Args:
          Fkernel: an array representing a convolution kernel
          axes: the convolution axes
        """
        
        self.Fkernel = Fkernel
        self.axes = axes
        if self.axes is None:
            self.axes = np.arange(len(self.Fkernel.shape))

    def frequency2real(self):
        """
        Converts the kernel from frequency space to real space with
        the origin shifted to the center.

        Returns:
          an array in real space          
        """
        return np.real_if_close(np.fft.fftshift(np.fft.ifftn(self.Fkernel,
                                axes=self.axes), axes=self.axes))

    def real2frequency(self, kernel):
        """
        Converts a kernel from real space to frequency space.

        Args:
          kernel: an array representing a convolution kernel in real space

        Returns:
          an array in frequency space
        """
        return np.fft.fftn(np.fft.ifftshift(kernel, axes=self.axes), axes=self.axes)

    def convolve(self, X):
        """
        Convolve X with a kernel in frequency space.

        Args:
          X: array to be convolved

        Returns:
          convolution of X with the kernel
        """
        if X.shape[1:] != self.Fkernel.shape:
            raise RuntimeError("Dimensions of X are incorrect.")
        FX = np.fft.fftn(X, axes=self.axes + 1)
        Fy = FX * self.Fkernel[None, ...]
        if len(self.axes) + 1 == len(self.Fkernel.shape):
            Fy = np.sum(Fy, axis=-1)
        return np.fft.ifftn(Fy, axes=self.axes + 1).real

    def resize(self, size):
        """
        Changes the size of the kernel to size.

        Args:
          size: tuple with the shape of the new kernel
        """
        if len(size) != len(self.Fkernel.shape):
            raise RuntimeError("length of resize shape is incorrect.")
        if not np.all(size >= self.Fkernel.shape[:-1]):
            raise RuntimeError("resize shape is too small.")

        kernel = self.frequency2real()
        padsize = np.array(size) - np.array(kernel.shape)
        paddown = padsize / 2
        padup = padsize - paddown
        padarray = np.concatenate((padup[..., None],
                                   paddown[..., None]), axis=1)
        pads = tuple([tuple(p) for p in padarray])
        kernel_pad = np.pad(kernel, pads, 'constant', constant_values=0)
        Fkernel_pad = self.real2frequency(kernel_pad)

        self.Fkernel = Fkernel_pad
