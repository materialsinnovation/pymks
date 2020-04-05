"""MKS Correlation Module
For computing auto and cross corelations under assumption
of  periodic or  non-periodic boundary conditions using discreete fourier
transform.


Note that input microstrucure should be 4 dimensional array.
where X=[n_sample,x,y.n_basis]
"""
import numpy as np
from toolz.curried import pipe, curry
from toolz.curried import map as map_, identity
from sklearn.base import TransformerMixin, BaseEstimator
import dask.array as da
from .func import dafftshift, dafftn, daifftn, daconj, flatten
from .func import sequence, make_da,zero_pad


def cross_correlation(arr1, arr2):
    """
    Returns the cross-correlation of and input field with another field.

    Args:
      arr1: the input field (n_samples,n_x,n_y)
      arr2: the other input field (n_samples,n_x,n_y)

    Returns:
      an nd-array of same dimension as the input field

    >>> import dask.array as da
    >>> x_data = np.asarray([[[1,1,0],
    ...                       [0,0,1],
    ...                       [1,1,0]]])
    >>> chunks = x_data.shape
    >>> x_data = da.from_array(x_data, chunks=chunks)
    >>> y_data = 1 - x_data
    >>> f_data = cross_correlation(x_data, y_data)
    >>> gg = np.asarray([[[ 2/9,  3/9,  2/9],
    ...                   [ 3/9, 0,  3/9],
    ...                   [ 2/9,  3/9,  2/9]]])
    >>> assert np.allclose(f_data.compute(), gg)
    >>> da.random.seed(42)
    >>> shape = (10, 5, 5)
    >>> chunks = (2, 5, 5)
    >>> x_data = da.random.random(shape, chunks=chunks)
    >>> y_data = 1 - x_data
    >>> f_data = cross_correlation(x_data, y_data)
    >>> assert x_data.chunks == f_data.chunks
    >>> shape = (10, 5, 5)
    >>> # When the two input fields have different chunkings
    >>> x_data = da.random.random(shape, chunks=(2,5,5))
    >>> y_data = da.random.random(shape, chunks=(5,5,5))
    >>> f_data = cross_correlation(x_data, y_data)
    >>> print(f_data.chunks)
    ((2, 2, 1, 1, 2, 2), (5,), (5,))
    """
    faxes = lambda x: tuple(np.arange(x.ndim - 1) + 1)

    return pipe(
        arr1,
        dafftn(axes=faxes(arr1)),
        lambda x: daconj(x) * dafftn(arr2, axes=faxes(arr2)),
        daifftn(axes=faxes(arr1)),
        dafftshift(axes=faxes(arr1)),
        lambda x: x.real / arr1[0].size,
    )


@curry
def auto_correlation(arr):
    """
    Returns auto-corrlation of and input field with itself.

    Args:
      arr: the input field (n_samples,n_x,n_y)

    Returns:
      an nd-array of same dimension as the input field

    >>> import dask.array as da
    >>> x_data = np.asarray([[[1, 1, 0],
    ...                       [0, 0, 1],
    ...                       [1, 1, 0]]])
    >>> chunks = x_data.shape
    >>> x_data = da.from_array(x_data, chunks=chunks)
    >>> f_data = auto_correlation(x_data)
    >>> gg = [[[3/9, 2/9, 3/9],
    ...        [2/9, 5/9, 2/9],
    ...        [3/9, 2/9, 3/9]]]
    >>> assert np.allclose(f_data.compute(), gg)
    >>> shape = (7, 5, 5)
    >>> chunks = (2, 5, 5)
    >>> da.random.seed(42)
    >>> x_data = da.random.random(shape, chunks=chunks)
    >>> f_data = auto_correlation(x_data)
    >>> assert x_data.chunks == f_data.chunks
    >>> print(f_data.chunks)
    ((2, 2, 2, 1), (5,), (5,))
    """
    return cross_correlation(arr, arr)


@curry
def center_slice(x_data, cutoff):
    """Calculate region of interest around the center voxel upto the
    cutoff length

    Args:
      x_data: the data array (n_samples,n_x,n_y), first index is left unchanged
      cutoff: cutoff size

    Returns:
      reduced size array

    >>> a = np.arange(7).reshape(1, 7)
    >>> print(center_slice(a, 2))
    [[1 2 3 4 5]]

    >>> a = np.arange(49).reshape(1, 7, 7)
    >>> print(center_slice(a, 1).shape)
    (1, 3, 3)

    >>> center_slice(np.arange(5), 1)
    Traceback (most recent call last):
    ...
    RuntimeError: Data should be greater than 1D

    """
    if x_data.ndim <= 1:
        raise RuntimeError("Data should be greater than 1D")

    make_slice = sequence(
        lambda x: x_data.shape[1:][x] // 2, lambda x: slice(x - cutoff, x + cutoff + 1)
    )

    return pipe(
        range(len(x_data.shape) - 1),
        map_(make_slice),
        tuple,
        lambda x: (slice(len(x_data)),) + x,
        lambda x: x_data[x],
    )


@curry
def two_point_stats(arr1, arr2, periodic_boundary=True, cutoff=None):
    """Calculate the 2-points stats for two arrays

    Args:
      arr1: array used to calculate cross-correlations (n_samples,n_x,n_y)
      arr2: array used to calculate cross-correlations (n_samples,n_x,n_y)
      periodic_boundary: whether to assume a periodic boundary (default is true)
      cutoff: the subarray of the 2 point stats to keep

    Returns:
      the snipped 2-points stats

    >>> two_point_stats(
    ...     da.from_array(np.arange(10).reshape(2, 5), chunks=(2, 5)),
    ...     da.from_array(np.arange(10).reshape(2, 5), chunks=(2, 5)),
    ... ).shape
    (2, 5)

    """
    if cutoff is None:
        # print("B",np.min(arr1.shape[1:]))
        cutoff = np.floor((np.min(arr1.shape[1:])-1)/2)
        # print(cutoff)
        # print("c",cutoff)
    # nonperiodic_padder = lambda x: np.pad(
    #     x,[(0,0)]+ [(cutoff, cutoff)] * (arr1.ndim-1), mode="constant", constant_values=0
    # ).rechunk((x.chunks[0], -1, -1, x.chunks[-1]))

    nonperiodic_padder =sequence(lambda x: np.pad(
        x,[(0,0)]+ [(cutoff, cutoff)] * (arr1.ndim-1), mode="constant", constant_values=0
    ),lambda x: x.rechunk(x.shape))

    if cutoff > np.floor((np.min(arr1.shape[1:])-1)/2):
         cutoff = np.floor((np.min(arr1.shape[1:])-1)/2)

    # nonperiodic_padder =
    # periodic_padder
    # print(arr1.ndim)
    padder = identity if periodic_boundary else nonperiodic_padder
    # print(padder(arr1).shape)
    # print(padder(arr1).compute())
    nonperiodic_normalize=lambda x: auto_correlation(padder(np.ones_like(x)))

    nonperiodic_stats= sequence(lambda x : cross_correlation(padder(x[0]), padder(x[1])),
        lambda x : x/nonperiodic_normalize(arr1),lambda x: center_slice(x,cutoff)
        )
    periodicstats=sequence(lambda x : cross_correlation(padder(x[0]), padder(x[1]))
        ,lambda x: center_slice(x,cutoff)
        )
    stats=periodicstats if periodic_boundary else nonperiodic_stats


    # normalize=identity if periodic_boundary else nonperiodic_normalize
    # stats =center_slice(cross_correlation(padder(arr1), padder(arr2)), cutoff)
    # print(nonperiodic_normalize(arr1).compute().shape)
    return stats([arr1,arr2])


@make_da
def correlations_multiple(data, correlations, periodic_boundary=True, cutoff=None):
    """Calculate 2-point stats for a multiple auto/cross correlation

    Args:
      data: the discretized data (n_samples,n_x,n_y,n_correlation)
      correlation_pair: the correlation pairs
      periodic_boundary: whether to assume a periodic boudnary (default is true)
      cutoff: the subarray of the 2 point stats to keep

    Returns:
      the 2-points stats array

    >>> data = np.arange(18).reshape(1, 3, 3, 2)
    >>> out = correlations_multiple(data, [[0, 1], [1, 1]])
    >>> out
    dask.array<stack, shape=(1, 3, 3, 2), dtype=float64, chunksize=(1, 3, 3, 1)>
    >>> answer = np.array([[[58, 62, 58], [94, 98, 94], [58, 62, 58]]]) + 1. / 3.
    >>> assert(out.compute()[...,0], answer)
    """

    return pipe(
        range(data.shape[-1]),
        map_(lambda x: (0, x)),
        lambda x: correlations if correlations else x,
        map_(
            lambda x: two_point_stats(
                data[..., x[0]],
                data[..., x[1]],
                periodic_boundary=periodic_boundary,
                cutoff=cutoff,
            )
        ),
        list,
        lambda x: da.stack(x, axis=-1),
    )


class TwoPointcorrelation(BaseEstimator, TransformerMixin):
    """Calculate the 2-point stats for two arrays
    """

    def __init__(self, periodic_boundary=True, cutoff=None, correlations=None):
        """Instantiate a TwoPointcorrelation

        Args:
          periodic_boundary: whether the boundary conditions are periodic
          cutoff: cutoff radius of interest for the 2PtStatistics field
          correlations1: an index
          correlations2: an index

        """

        self.correlations = correlations
        self.periodic_boundary = periodic_boundary
        self.cutoff = cutoff

    def transform(self, data):
        """Transform the data

        Args:
          data: the data to be transformed

        Returns:
          the 2-point stats array
        """
        return correlations_multiple(
            data,
            self.correlations,
            periodic_boundary=self.periodic_boundary,
            cutoff=self.cutoff,
        )

    def fit(self, *_):
        """Only necessary to make pipelines work
        """
        return self


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Reshape data ready for the Principle Component Analysis

    Two point correlation data need to be flatten before performing
    Principle component Analysis.This class flattens the TwoPoint correlation
    data for scikit learn pipeline

    >>> data = np.arange(50).reshape((2, 5, 5))
    >>> FlattenTransformer().transform(data).shape
    (2, 25)

    """

    @staticmethod
    def transform(x_data):
        """Transform the X data

        Args:
            x_data: the data to be transformed
        """
        return flatten(x_data)

    def fit(self, *_):
        """Only necessary to make pipelines work
        """
        return self
