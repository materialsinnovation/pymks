import numpy as np
import scipy as sc


def Legendre(X, deg, domain):
    '''This method takes in a continuos microstructure and returns a
    microstructure function with a Legendrian Polynomial as the basis.

    Args:
      X: Microstructure
      deg: The deg of Legendre polynomials used in the microstructure function.
      domain: The domain for the microstructure.
    Returns:
      Legendre polynomials up to deg evalutated at value of X for each cell.

    >>> X = np.array([[0.25, 0.1],
    ...               [0.5, 0.25]])
    >>> X_Legendre = np.array([[[-0.3125, -0.75, 0.5],
    ...                         [ 1.15,   -1.2, 0.5]],
    ...                        [[-1.25,      0, 0.5],
    ...                         [-0.3125, -0.75, 0.5]]])
    >>> assert(np.allclose(Legendre(X, 3, [0., 0.5]), X_Legendre))
    '''
    leg = np.polynomial.legendre
    X_scaled = 2. * X - domain[0] - domain[1] / (domain[1] - domain[0])
    norm = (2. * np.arange(deg) + 1) / 2.
    X_Legendre = np.flipud(leg.legval(X_scaled, np.eye(deg) * norm))
    return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))


def Legendre_fft(X, deg, domain):
    X_Legendre = Legendre(X, deg, domain)
    axis = np.arange(len(X.shape) - 1) + 1
    return np.fft.fftn(X_Legendre, axes=axis)


def Hermite(X, deg, **kargs):
    '''This method takes in a continous microstructure and returns a
    microstructure function with a Hermite Polynomial basis.

    Args:
      X: Microstructure
      deg: The deg of Hermite polynomials used in the microstructure function.
    Returns:
      Hermite polynomial up to deg evaluated at the value of X for each cell.

    >>> X = np.array([[0.25, 0.],
    ...               [1., 0.25]])
    >>> X_Hermite = np.array([[[-0.11593905, 0.13250177, 0.53000706],
    ...                         [-0.1410474 , 0.        , 0.56418958]],
    ...                        [[ 0.05188844, 0.20755375, 0.20755375],
    ...                         [-0.11593905, 0.13250177, 0.53000706]]])
    >>> assert(np.allclose(Hermite(X, 3), X_Hermite))
    '''

    herm = np.polynomial.hermite
    X_weight = herm.hermweight(X).repeat(deg, axis=1).reshape(X.shape + (deg,))
    n = np.arange(deg)
    norm = (np.sqrt(np.pi) * sc.misc.factorial(n) * 2 ** n) ** -1
    X_Hermite = np.flipud(herm.hermval(X, np.eye(deg) * norm))
    return np.rollaxis(X_Hermite, 0, len(X_Hermite.shape)) * X_weight


def Hermite_fft(X, deg):
    X_Hermite = Hermite(X, deg)
    axis = np.arange(len(X.shape) - 1) + 1
    return np.fft.fftn(X_Hermite, axes=axis)
