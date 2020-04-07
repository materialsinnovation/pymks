"""MKS localzation module

The MKS localization module fits data using the Materials Knowledge
System in Fourier Space.

Example:

>>> from pymks.fmks.bases.primitive import discretize
>>> from pymks.fmks.func import allclose

>>> disc = discretize(n_state=2)
>>> x_data = lambda: da.from_array(np.linspace(0, 1, 8).reshape((2, 2, 2)),
...                                chunks=(1, 2, 2))
>>> y_data = lambda: x_data().swapaxes(1, 2)
>>> assert pipe(
...     fit(x_data(), y_data(), disc),
...     predict(x_data(), discretize=disc),
...     allclose(y_data())
... )
"""

import dask.array as da
import numpy as np
from scipy.linalg import lstsq
from toolz.curried import pipe
from toolz.curried import map as fmap
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator

from .func import (
    curry,
    array_from_tuple,
    rechunk,
    dafftshift,
    dafftn,
    daifftn,
    daifftshift,
    zero_pad,
)


@curry
def lstsq_mks(fx_data, fy_data, redundancy_func, ijk):
    """Do a least square for a single point in the MKS k-space.

    Select a point in k-space using `ijk` tuple and do a least squares
    across samples and local state space to calculate parts of the
    coefficient matrix.

    Args:
      fx_data: microstructure in k-space
      fy_data: response in k-space
      redundancy_func: helps remove redundancies in the coefficient matrix
      ijk: a point in k-space

    Returns:
      a tuple of the coefficient matrix index and vector over local
      state space, which can be used to populate the coefficient
      matrix


    >>> make_data = lambda s: da.from_array(np.arange(np.prod(s),
    ...                                               dtype=float).reshape(s),
    ...                                     chunks=s)

    >>> index, value = lstsq_mks(make_data((2, 1, 1, 3)),
    ...                          make_data((2, 1, 1)),
    ...                          lambda _: (slice(None),),
    ...                          (0, 0))

    >>> print(index)
    (0, 0, slice(None, None, None))
    >>> assert np.allclose(value, [5. / 18., 1. / 9., -1. / 18.])

    """
    fx_data_ = lambda: fx_data[(slice(None),) + ijk + redundancy_func(ijk)]
    fy_data_ = lambda: fy_data[(slice(None),) + ijk]
    return (
        ijk + redundancy_func(ijk),
        lstsq(fx_data_(), fy_data_(), np.finfo(float).eps * 1e4)[0],
    )


def fit_fourier(fx_data, fy_data, redundancy_func):
    """Fit the data in Fourier space.

    Fit the data after it has been discretized and transformed into
    Fourier space.

    Args:
      fx_data: microstructure in k-space
      fy_data: response in k-space
      redundancy_func: helps remove redundancies in the coefficient matrix

    Returns:
      the coefficient matrix (unchunked)

    >>> make_data = lambda s: da.from_array(np.arange(np.prod(s),
    ...                                               dtype=float).reshape(s),
    ...                                     chunks=s)

    >>> matrix = fit_fourier(make_data((5, 4, 4, 3)),
    ...                      make_data((5, 4, 4)),
    ...                      lambda _: (slice(None),))

    >>> print(matrix.shape)
    (4, 4, 3)

    >>> test_matrix = np.resize([5. / 18., 1. / 9., -1. / 18.], (4, 4, 3))
    >>> assert np.allclose(matrix, test_matrix)

    """
    lstsq_mks_ = lstsq_mks(fx_data.compute(), fy_data.compute(), redundancy_func)
    return pipe(
        fmap(lstsq_mks_, np.ndindex(fx_data.shape[1:-1])),
        list,
        array_from_tuple(shape=fx_data.shape[1:], dtype=np.complex),
    )


def faxes(arr):
    """Get the spatiol axes to perform the Fourier transform

    The first and the last axes should not have the Fourier transform
    performed.

    Args:
      arr: the discretized array

    Returns:
      an array starting at 1 to n - 2 where n is the length of the
      shape of arr

    >>> faxes(np.array([1]).reshape((1, 1, 1, 1, 1)))
    array([1, 2, 3])

    """
    return np.arange(arr.ndim - 2) + 1


@curry
def fit_disc(x_data, y_data, redundancy_func):
    """Fit the discretized data.

    Fit the data after the data has already been discretized.

    Args:
      x_data: the discretized mircrostructure field
      y_data: the discretized response field
      redundancy_func: helps remove redundancies in the coefficient matrix

    Returns:
      the chunked coefficient matrix based on the chunking of local
      state space from the discretized mircrostructure field


    >>> make_data = lambda s, c: da.from_array(
    ...     np.arange(np.prod(s),
    ...               dtype=float).reshape(s),
    ...     chunks=c
    ... )

    >>> matrix = fit_disc(make_data((6, 4, 4, 3), (2, 4, 4, 1)),
    ...                   make_data((6, 4, 4), (2, 4, 4)),
    ...                   lambda _: (slice(None),))

    >>> print(matrix.shape)
    (4, 4, 3)

    >>> print(matrix.chunks)
    ((4,), (4,), (1, 1, 1))

    >>> assert np.allclose(matrix.compute()[0, 0, 0], 5. / 18.)

    """
    chunks = lambda x: (None,) * (len(x.shape) - 1) + (x_data.chunks[-1],)
    return pipe(
        [x_data, y_data],
        fmap(dafftn(axes=faxes(x_data))),
        list,
        lambda x: fit_fourier(*x, redundancy_func),
        lambda x: da.from_array(x, chunks=chunks(x)),
    )


@curry
def fit(x_data, y_data, discretize, redundancy_func=lambda _: (slice(None),)):
    """Calculate the MKS influence coefficients.

    Args:
      x_data: the microstructure field
      y_data: the response field
      discretize: a function that returns the discretized data and
        redundancy function


    Returns:
      the influence coefficients

    >>> from pymks.fmks.bases.primitive import discretize

    >>> matrix = fit(da.from_array(np.array([[0], [1]]), chunks=(2, 1)),
    ...              da.from_array(np.array([[2], [1]]), chunks=(2, 1)),
    ...              discretize(n_state=3))
    >>> assert np.allclose(matrix, [[2, 0, 1]])

    """
    return pipe(
        x_data, discretize, fit_disc(y_data=y_data, redundancy_func=redundancy_func)
    )


@curry
def _predict_disc(x_data, coeff):
    return pipe(
        dafftn(x_data, axes=faxes(x_data)),
        lambda x: np.sum(x * coeff[None], axis=-1),
        daifftn(axes=faxes(x_data), s=x_data.shape[1:-1]),
    ).real


@curry
def predict(x_data, coeff, discretize):
    """Predict a response given a microstructure

    Args:
      x_data: the microstructure data
      coeff: the influence coefficients
      discretize: the basis function

    Returns:
      the response
    """
    return _predict_disc(discretize(x_data), coeff)


def _ini_axes(arr):
    return tuple(np.arange(arr.ndim - 1))


@curry
def coeff_to_real(coeff, new_shape=None):
    """Covert the coefficents to real space

    Args:
      coeff: the coefficient in Fourier space
      new_shape: the shape of the coefficients in real space

    Returns:
      the coefficients in real space
    """
    return pipe(
        coeff,
        daifftn(axes=_ini_axes(coeff), s=new_shape),
        dafftshift(axes=_ini_axes(coeff)),
    )


@curry
def coeff_to_frequency(coeff):
    """Convert the coefficients to frequency space.

    Args:
      coeff: the influence coefficients in real space

    Returns:
      the influence coefficiencts in frequency space

    >>> from .func import rcompose
    >>> f = rcompose(
    ...     lambda x: np.concatenate((x, np.ones_like(x)), axis=-1),
    ...     lambda x: da.from_array(x, chunks=x.shape),
    ...     coeff_to_frequency,
    ...     coeff_to_real,
    ...     lambda x: x.real[..., :1].compute()
    ... )
    >>> assert (lambda x: np.allclose(f(x), x))(np.arange(20).reshape((5, 4, 1)))

    """
    return pipe(
        coeff.copy(), daifftshift(axes=_ini_axes(coeff)), dafftn(axes=_ini_axes(coeff))
    )


@curry
def coeff_resize(coeff, shape):
    """Resize the influence coefficients.

    Resize the influence coefficients by padding with zeros to the
    size determined by shape. Apply to coefficients in frequency space.

    Args:
      coeff: the influence coefficients with size (nx, ny, nz, nstate)
      shape: the new padded shape (NX, NY, NZ)

    Returns:
      the resized influence coefficients

    >>> from .func import ifftshift, fftn
    >>> assert pipe(
    ...     np.arange(20).reshape((5, 4, 1)),
    ...     lambda x: np.concatenate((x, np.ones_like(x)), axis=-1),
    ...     ifftshift(axes=(0, 1)),
    ...     fftn(axes=(0, 1)),
    ...     lambda x: da.from_array(x, chunks=x.shape),
    ...     coeff_resize(shape=(10, 7)),
    ...     coeff_to_real,
    ...     lambda x: np.allclose(x.real[..., 0],
    ...         [[0, 0, 0, 0, 0, 0, 0],
    ...          [0, 0, 0, 0, 0, 0, 0],
    ...          [0, 0, 0, 0, 0, 0, 0],
    ...          [0, 0, 0, 1, 2, 3, 0],
    ...          [0, 0, 4, 5, 6, 7, 0],
    ...          [0, 0, 8, 9,10,11, 0],
    ...          [0, 0,12,13,14,15, 0],
    ...          [0, 0,16,17,18,19, 0],
    ...          [0, 0, 0, 0, 0, 0, 0],
    ...          [0, 0, 0, 0, 0, 0, 0]])
    ... )

    """
    return pipe(
        coeff,
        coeff_to_real,
        zero_pad(
            shape=shape + coeff.shape[-1:],
            chunks=((-1,) * len(shape)) + (coeff.chunks[-1],),
        ),
        coeff_to_frequency,
    )


def reshape(data, shape):
    """Reshape data along all but the first axis

    Args:
        data: the data to reshape
        shape: the shape of the new data (not including the first axis)

    Returns:
        the reshaped data

    >>> data = np.arange(18).reshape((2, 9))
    >>> reshape(data, (None, 3, 3)).shape
    (2, 3, 3)
    """
    return data.reshape(data.shape[0], *shape[1:])


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    """Reshape data ready for the LocalizationRegressor

    Sklearn likes flat image data, but MKS expects shaped data. This
    class transforms the shape of flat data into shaped image data for
    MKS.

    Attributes:
       shape: the shape of the reshaped data (ignoring the first axis)

    >>> data = np.arange(18).reshape((2, 9))
    >>> ReshapeTransformer((None, 3, 3)).fit(None, None).transform(data).shape
    (2, 3, 3)

    """

    def __init__(self, shape):
        """Instantiate a ReshapeTransformer

        Args:
            shape: the shape of the reshaped data (ignoring the first axis)
        """
        self.shape = shape

    def transform(self, x_data):
        """Transform the X data

        Args:
            x_data: the data to be transformed
        """
        return reshape(x_data, self.shape)

    def fit(self, *_):
        """Only necessary to make pipelines work
        """
        return self


class LocalizationRegressor(BaseEstimator, RegressorMixin):
    """Perform the localization in Sklearn pipelines

    Allows the localization to be part of a Sklearn pipeline

    Attributes:
        redundancy_func: function to remove redundant elements from
            the coefficient matrix
        coeff: the coefficient matrix
        y_data_shape: the shape of the predicited data

    >>> make_data = lambda s, c: da.from_array(
    ...     np.arange(np.prod(s),
    ...               dtype=float).reshape(s),
    ...     chunks=c
    ... )

    >>> X = make_data((6, 4, 4, 3), (2, 4, 4, 1))
    >>> y = make_data((6, 4, 4), (2, 4, 4))

    >>> y_out = LocalizationRegressor().fit(X, y).predict(X)

    >>> assert np.allclose(y, y_out)

    >>> print(
    ...     pipe(
    ...         LocalizationRegressor(),
    ...         lambda x: x.fit(X, y.reshape(6, 16)).predict(X).shape
    ...     )
    ... )
    (6, 16)

    """

    def __init__(self, redundancy_func=lambda _: (slice(None),)):
        """Instantiate a LocalizationRegressor

        Args:
            redundancy_func: function to remove redundant elements
                from the coefficient matrix
        """
        self.redundancy_func = redundancy_func
        self.coeff = None
        self.y_data_shape = None

    def fit(self, x_data, y_data):
        """Fit the data

        Args:
            x_data: the X data to fit
            y_data: the y data to fit

        Returns:
            the fitted LocalizationRegressor
        """
        self.y_data_shape = y_data.shape
        y_data_reshape = reshape(y_data, x_data.shape[:-1])
        y_data_da = rechunk(y_data_reshape, chunks=x_data.chunks[:-1])
        self.coeff = fit_disc(x_data, y_data_da, self.redundancy_func)
        return self

    def predict(self, x_data):
        """Predict the data

        Args:
            x_data: the X data to predict

        Returns:
            The predicted y data
        """
        if len(self.y_data_shape) == len(x_data.shape) - 1:
            new_shape = (1,) + self.coeff.shape[:-1]
        else:
            new_shape = (1, np.prod(self.coeff.shape[:-1]))
        return reshape(_predict_disc(x_data, self.coeff), new_shape)

    def coeff_resize(self, shape):
        """Generate new model with larger coefficients

        Args:
          shape: the shape of the new coefficients

        Returns:
          a new model with larger influence coefficients
        """
        self.coeff = coeff_resize(self.coeff, shape)
        return self
