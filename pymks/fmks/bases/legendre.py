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

>>> data = 2 * da.random.random((1, 3, 3), chunks=(1, 3, 3)) - 1
>>> assert(data.shape == (1, 3, 3))

The when a microstructure is discretized, the different local states are
mapped into local state space, which results in an array of shape
`(n_samples, n_x, n_y, n_states)`, where `n_states=4` in this case.

>>> from toolz import pipe
>>> data_ = pipe(data, discretize(n_state=4, min_=-1, max_=1))
>>> assert(data_.shape == (1, 3, 3, 4))

"""


import numpy as np
import numpy.polynomial.legendre as leg
import dask.array as da
from toolz.curried import pipe
from ..func import curry
from .basis_transformer import BasisTransformer


@curry
def scaled_data(data, domain):
    """
    Scales data to range between -1.0 and 1.0, viz. the domain over which
    legendre polynomials are defined
    """
    return (2.0 * data - domain[0] - domain[1]) / (domain[1] - domain[0])


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
    return da.map_blocks(
        lambda x: leg.legval(x, coeff_, tensor=False),
        data,
        chunks=data.chunks[:-1] + (len(coeff_),),
    )


@curry
def discretize(data, n_state=2, min_=0, max_=1, chunks=None):
    """legendre discretization of a microstructure.

    Args:
      data: The microstructure as an `(n_samples, n_x, ...)` shaped
        array where `n_samples` is the number of samples and `n_x` is
        the spatial discretization.
      n_state: the number of local states
      min_: the minimum local state
      max_: the maximum local state
      chunks: the chunks size for the state axis

    Returns:
      Float valued field of of Legendre polynomial coefficients

    >>> data = da.from_array(np.arange(4).reshape(2, 2), chunks=(2, 2))
    >>> out = discretize(data, n_state=3, max_=3, chunks=2)
    >>> print(out.shape)
    (2, 2, 3)
    >>> print(out.chunks)
    ((2,), (2,), (2, 1))
    """
    return pipe(
        data[..., None],
        scaled_data(domain=(min_, max_)),
        leg_data(coeff_=coeff(np.arange(n_state))),
        lambda x: x.rechunk(data.chunks + (chunks or n_state,)),
    )


class LegendreTransformer(BasisTransformer):
    """Legendre transformer for Sklearn pipelines

    >>> from toolz import pipe
    >>> data = da.from_array(np.array([[0, 0.5, 1]]), chunks=(1, 3))
    >>> pipe(
    ...     LegendreTransformer(),
    ...     lambda x: x.fit(None, None),
    ...     lambda x: x.transform(data).compute(),
    ... )
    array([[[ 0.5, -1.5],
            [ 0.5,  0. ],
            [ 0.5,  1.5]]])

    """

    def __init__(self, n_state=2, min_=0.0, max_=1.0, chunks=None):
        """Instantiate a LegendreTransformer

        Args:
          n_state: the number of local states
          min_: the minimum local state
          max_: the maximum local state
          chunks: chunks size for state axis
        """
        super().__init__(
            discretize, n_state=n_state, min_=min_, max_=max_, chunks=chunks
        )
