import numpy as np


def get_microstructure(n_samples, size, n_phases):
    '''
    Generates n_samples number of a periodic random microstructure with size
    equal to size and with n_phases number of phases.

    >>> from pymks.datasets.microstructureGenerator import get_microstructure
    >>> np.random.seed(10)
    >>> n_samples, n_phases = 1, 2
    >>> size = (3, 3)
    >>> X = get_microstructure(n_samples, size, n_phases)
    >>> X_test = np.array([[[0, 1, 0],
    ...                     [0, 0, 1],
    ...                     [1, 0, 1]]])
    >>> assert(np.allclose(X, X_test))

    Args:
      n_samples: number of samples to be genrated
      size: size of samples
      n_phases: number of phases in microstructures.

    Returns:
      n_samples number of a periodic random microstructure with size equal to
      size and with n_phases number of phases.
    '''
    X = np.random.random((n_samples,) + size)
    gaussian_filter = _make_filter(size)
    gaussian_filter = np.tile(gaussian_filter, (n_samples,) +
                              tuple(np.ones(len(gaussian_filter.shape),
                                            dtype=int)))
    X_blur = _convolve_fft(X, gaussian_filter,
                           axes=np.arange(len(X[0].shape)) + 1)
    return _assign_phases(X_blur, n_phases)


def _make_filter(size):
    '''
    Create filter for convolution.
    '''
    M = np.max(size) / 2
    gaussian = np.exp(-((np.arange(M) - (M / 2)) / (M / 8.)) ** 2)
    dim = len(size)
    gaussian = np.ones((np.array(size)) / 2) * gaussian
    gaussian_tmp = np.ones((dim,) + gaussian.shape)
    for ii in range(dim):
        gaussian = np.rot90(gaussian.swapaxes(0, dim - 1), dim)
        gaussian_tmp[ii] = gaussian
    gaussian_filter = np.prod(gaussian_tmp, axis=0)
    pads = tuple([(0, ((p + 1) / 2)) for p in size])
    return np.pad(gaussian_filter, pads, 'constant', constant_values=0)


def _convolve_fft(X, filt, axes):
    '''
    Convolve X and the filter using FFT method.
    '''
    FX = np.fft.fftn(X, axes=axes)
    Ffilt = np.fft.fftn(filt, axes=axes)
    return np.fft.ifftn(FX * np.conj(Ffilt), axes=axes).real


def _assign_phases(X, n_phases):
    '''
    Takes in blured array and assigns phase values.
    '''
    x = np.linspace(np.min(X), np.max(X), n_phases + 1)[:-1][::-1]
    for ii in range(len(x)):
        X = np.where(X >= x[ii], -int(ii), X)
    return np.intc(-X)
