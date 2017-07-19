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
>>> n_states = 3
>>> X = np.array([[0.25, 0.1],
...               [0.5, 0.25]])
>>> def P(x):
...    x = 4 * x - 1
...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
...    return np.rollaxis(tmp, 0, 3)
>>> domain = [0., 0.5]
>>> chunks = (1,)
>>> assert(np.allclose(legendre_basis(X, domain, n_states, chunks)[0].compute(), P(X)))
"""

import numpy as np
import numpy.polynomial.legendre as leg
import dask.array as da


def scaled_x(X, domain):
    return ((2.*X-domain[0]-domain[1])/(domain[1]-domain[0]))


def norm(n_states):
    return (2.*np.array(n_states)+1)/2.


def coeff(n_states):
    return np.eye(len(n_states))*norm(n_states)


def leg_x(X, domain, n_states):
     return (leg.legval(scaled_x(X, domain), coeff(n_states)))


def rollaxis_(X):
    return np.rollaxis(X, 0, len(X.shape))


def redundancy(ijk):
    return (slice(-1),)


def discretize(X, domain=[-1,1], n_states=np.arange(2), chunks=()):
        return rollaxis_(leg_x(X, domain, n_states))


def is_in_domain(X, domain):
    return ((np.min(X) < domain[0]) or (np.max(X) > domain[1]))


def legendre_basis(X, domain=[-1,1], n_states=2, chunks=(1,)):
    return (da.asarray(discretize(np.asarray(X), domain,
	np.arange(n_states))).rechunk(chunks=X.shape+chunks),
	redundancy)
