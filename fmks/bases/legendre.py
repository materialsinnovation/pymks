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
>>> assert(np.allclose(legendre_basis(X,
...                    n_state,
...                    domain,
...                    chunks)[0].compute(), P(X)))
"""

import numpy as np
import numpy.polynomial.legendre as leg
import dask.array as da
from toolz.curried import pipe
from ..func import curry


@curry
def scaled_data(data, domain):
    """Sclaes data to range between -1.0 and 1.0"""
    return (2. * data - domain[0] - domain[1]) / (domain[1] - domain[0])


@curry
def coeff(states):
    """returns coefficients for input as parameters to legendre value a
    calculations"""
    return np.eye(len(states)) * (states + 0.5)


@curry
def leg_data(data, coeff_):
    """Computes legendre expansion for each data point in the
    input data matrix.
    """
    return leg.legval(data, coeff_)


@curry
def discretize(data, states=np.arange(2), domain=(0, 1)):
    """legendre discretization of a microstructure.

    Args:
        x_data (ND array) : The microstructure as an `(n_samples, n_x, ...)`
            shaped array where `n_samples` is the number of samples and
            `n_x` is the spatial discretization.
        n_state (ND array)    : rangle of local states.
        domain  (float tuple) : the minimum and maximum range for local states

    Returns:
        Float valued field of of Legendre polynomial coefficients as a
        numpy array.
    """
    return pipe(data[..., None],
                scaled_data(domain=domain),
                leg_data(coeff_=coeff(states)))


@curry
def legendre_basis(x_data, n_state=2, domain=(0, 1), chunks=(1,)):
    """legendre discretization of a microstructure.

    Args:
        x_data (ND array) : The microstructure as an `(n_samples, n_x, ...)`
            shaped array where `n_samples` is the number of samples and
            `n_x` is the spatial discretization.
        n_state (float)       : the number of local states
        domain  (float tuple) : the minimum and maximum range for local states

    Returns:
        Float valued field of of Legendre polynomial coefficients as a chunked
        dask array.
    >>> # test1def discretize(data, n_state=np.arange(2), domain=(0, 1)):
    >>> X = np.array([[-1, 1],
    ...               [0, -1]])
    >>> leg_basis = legendre_basis(n_state=3, domain=(-1, 1))
    >>> def p(x):
    ...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
    ...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
    ...    return np.rollaxis(tmp, 0, 3)
    >>> assert(np.allclose(leg_basis(X)[0].compute(), p(X)))
    >>> # test 2
    >>> np.random.seed(3)
    >>> X = np.random.random((1, 3, 3))
    >>> leg_basis = legendre_basis(n_state=2, domain=(0, 1))
    >>> X_ = leg_basis(X)[0].compute()
    >>> FX = np.fft.fftn(X_, axes=(1, 2))
    >>> FXtest = np.array([[[[4.50000000+0.j, -0.79735949+0.],
    ...                      [0.00000000+0.j, -1.00887157-1.48005289j],
    ...                      [0.00000000+0.j, -1.00887157+1.48005289j]],
    ...                     [[0.00000000+0.j, 0.62300683-4.97732233j],
    ...                      [0.00000000+0.j, 1.09318216+0.10131035j],
    ...                      [0.00000000+0.j, 0.37713401+1.87334545j]],
    ...                     [[0.00000000+0.j, 0.62300683+4.97732233j],
    ...                      [0.00000000+0.j, 0.37713401-1.87334545j],
    ...                      [0.00000000+0.j, 1.09318216-0.10131035j]]]])
    >>> assert np.allclose(FX, FXtest)
    """
    return (discretize(da.from_array(x_data,
                                     chunks=chunks + x_data.shape[1:]),
                       np.arange(n_state),
                       domain),
            lambda x: (slice(-1),))
