r"""
Legendre basis for microstructure discretization.

Discretize the microstructure function into `n_states` local states
such that:

.. math::
   \frac{1}{\Delta x} \int_s m(h, x) dx =
   \sum_0^{L-1} m[l, s] P_l(h)

where the :math:`P_l` are Legendre polynomials and the local state space
:math:`H` is mapped into the orthogonal domain of the Legendre polynomials

.. math::
   -1 \le  H \le 1

Example.

Here is an example with 4 local states in a microstructure.

>>> data = 2 * np.random.random((1, 3, 3)) - 1
>>> assert(data.shape == (1, 3, 3))

The when a microstructure is discretized, the different local states are
mapped into local state space, which results in an array of shape
`(n_samples, n_x, n_y, n_states)`, where `n_states=4` in this case.

>>> from toolz import pipe
>>> data_ = pipe(data, legendre_basis(n_state=4, domain=(-1,1)))
>>> assert(data_[0].shape == (1, 3, 3, 4))

"""


import numpy as np
import numpy.polynomial.legendre as leg
import dask.array as da
from toolz.curried import pipe
from ..func import curry


@curry
def scaled_data(data, domain):
    """
    Scales data to range between -1.0 and 1.0, viz. the domain over which
    legendre polynomials are defined
    """
    return (2. * data - domain[0] - domain[1]) / (domain[1] - domain[0])


@curry
def coeff(states):
    """
    Returns a diagonal matrix where in the diagonal terms serve as
    coefficients for individual terms in Legendre polynomial expansion.
    """
    return np.eye(len(states)) * (states + 0.5)


@curry
def leg_data(data, coeff_):
    """
    Computes Legendre expansion for each data point in the
    input data matrix.
    """
    return leg.legval(data, coeff_)


@curry
def discretize(data, states=np.arange(2), domain=(0, 1)):
    """
    legendre discretization of a microstructure.

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
    return pipe(
        data[..., None], scaled_data(domain=domain), leg_data(coeff_=coeff(states))
    )


@curry
def legendre_basis(x_data, n_state=2, domain=(0, 1), chunks=(1,)):
    """
    legendre discretization of a microstructure.

    Args:
        x_data (ND array) : The microstructure as an `(n_samples, n_x, ...)`
            shaped array where `n_samples` is the number of samples and
            `n_x` is the spatial discretization.
        n_state (float)       : the number of local states
        domain  (float tuple) : the minimum and maximum range for local states

    Returns:
        Float valued field of of Legendre polynomial coefficients as a chunked
        dask array of shape `(n_samples, n_x, ..., n_state)`.
    """
    return (
        discretize(
            da.from_array(x_data, chunks=chunks + x_data.shape[1:]),
            np.arange(n_state),
            domain,
        ),
        lambda x: (slice(-1),),
    )
