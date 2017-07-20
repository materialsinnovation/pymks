r"""
Discretize a continuous field into `deg` local states using a
Legendre polynomial basis such that,
.. math::
   \frac{1}{\Delta x} \int_s m(h, x) dx =
   \sum_0^{L-1} m[l, s] P_l(h)
where the :math:`P_l` are Legendre polynomials and the local state space
:math:`H` is mapped into the orthogonal domain of the Legendre polynomials
.. math::
   -1 \le  H \le 1
The mapping of :math:`H` into the domain is done automatically in PyMKS by
using the `domain` key work argument.
>>> n_state = 3
>>> X = np.array([[0.25, 0.1],
...               [0.5, 0.25]])
>>> def P(x):
...    x = 4 * x - 1
...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
...    return np.rollaxis(tmp, 0, 3)
>>> domain = [0., 0.5]
>>> chunks = (1,)
>>> assert(np.allclose(legendre_basis(X, n_state, domain, chunks)[0].compute(), P(X)))
"""

import numpy as np
import numpy.polynomial.legendre as leg
import dask.array as da
from ..func import curry

@curry
def scaled_data(data, domain):
    """Sclaes data to range between -1.0 and 1.0"""
    return (2.*data-domain[0]-domain[1])/(domain[1]-domain[0])

@curry
def norm(n_state):
    """returns normalized local states"""
    return (2.*np.array(n_state)+1)/2.

@curry
def coeff(n_state):
    """returns coefficients for input as parameters to legendre value a
    calculations"""
    return np.eye(len(n_state))*norm(n_state)

@curry
def leg_data(data, domain, n_state):
    """computes legendre expansion for the data"""
    return leg.legval(scaled_data(data, domain),
                      coeff(n_state))

@curry
def rollaxis_(data):
    """shifts the discretised axis to the back of the stack"""
    return np.rollaxis(data, 0, len(data.shape))


def redundancy(ijk):
    """Used in localization to remove redundant slices in
    case of primitive basis function. Not used elsewhere.

    Args:
      ijk: the current index

    Returns:
      the redundant slice, (slice(-1),) when no redundancies
    """
    if np.all(np.array(ijk) == 0):
        return (slice(-1),)
    return (slice(-1),)

@curry
def discretize(data, n_state=np.arange(2), domain=(0, 1)):
    """descretizes the data"""
    return rollaxis_(leg_data(data, domain, n_state))


@curry
def legendre_basis(x_data, n_state=2, domain=(0, 1), chunks=(1,)):
    """the user accessible api"""
    return (da.asarray(discretize(np.asarray(x_data),
                                  np.arange(n_state),
                                  domain)
                      ).rechunk(chunks=x_data.shape+chunks),
            redundancy)
