import numpy as np


def Legendre(X, deg, domain):
    '''This method takes in a continuos microstructure and returns a
    microstructure	function with Legendrian Polynomial as the basis.

    Args:
      X: Microstructure
      deg: The deg of Legendre Polynomials used in the microstructure function.
      domain: The domain for the microstructure.
    Returns:
      Legendre polynomials up to deg evalutated at value of X for each cell.

    >>> X = np.array([[0.25, 0.1],
    ...               [0.5, 0.25]])
    >>> X_Legendre = np.array([[[-0.625, -1.5, 1.],
    ...                         [ 2.3,   -2.4, 1.]],
    ...                        [[-2.5,      0, 1.],
    ...                         [-0.625, -1.5, 1.]]])
    >>> assert(np.allclose(Legendre(X, 3, [0., 1.]), X_Legendre))
    '''
    leg = np.polynomial.legendre
    X_scaled = 2 * X - domain[0] - domain[1] / (domain[1] - domain[0])
    norm = 2 * np.arange(deg) + 1
    X_Legendre = np.flipud(leg.legval(X_scaled, np.eye(deg) * norm))
    return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))


def Legendre_fft(X, deg, domain):
    X_Legendre = Legendre(X, deg, domain)
    axis = np.arange(len(X.shape) - 1) + 1
    return np.fft.fftn(X_Legendre, axes=axis)
