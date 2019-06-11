"""MKS Correlation Module
For computing auto and cross corelations under assumption
of  periodic or  non-periodic boundary conditions using discreete fourier
transform.


Note that input microstrucure should be 4 dimensional array.
where X=[n_sample,x,y.n_basis]
"""
import numpy as np
from toolz.curried import pipe, curry
from sklearn.base import TransformerMixin, BaseEstimator
import dask.array as da
from .func import dafftshift, dafftn, daifftn, daconj


def faxes(arr):
    """Get the spatial axes to perform the Fourier transform

    The first axis should not have the Fourier transform
    performed.

    Args:
      arr: the discretized array
    Returns:
      an array starting at 1 to n - 1 where n is the length of the
      shape of arr

    >>> faxes(np.array([1]).reshape((1, 1, 1, 1, 1)))
    (1, 2, 3, 4)
    """
    return tuple(np.arange(arr.ndim - 1) + 1)


def corr_master(arr1, arr2):
    """
    Returns cross correlation between the two input fields, arr1 and arr2
    """
    return pipe(
        arr1,
        dafftn(axes=faxes(arr1)),
        lambda x: daconj(x) * dafftn(arr2, axes=faxes(arr2)),
        daifftn(axes=faxes(arr1)),
        dafftshift(axes=faxes(arr1)),
        lambda x: x.real,
    )


@curry
def auto_correlation(arr1):
    """
    Returns auto-corrlation of and input field with itself.
    Args:
        arr1: the input field

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
    return corr_master(arr1, arr1) / arr1[0].size


@curry
def cross_correlation(arr1, arr2):
    """
    Returns the cross-correlation of and input field with another field.
    Args:
        arr1: the input field
        arr2: the other input field

    Returns:
        an nd-array of same dimension as the input field

    >>> import dask.array as da
    >>> x_data = np.asarray([[[1,1,0],
    ...                       [0,0,1],
    ...                       [1,1,0]]])
    >>> chunks = x_data.shape
    >>> x_data = da.from_array(x_data, chunks=chunks)
    >>> y_data = da.from_array(1 - x_data, chunks=chunks)
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

    # Check if normalization is correct
    return corr_master(arr1, arr2) / arr1[0].size


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


def flatten(data):
    """Flatten data along all but the first axis

    Args:
        data: data to flatten

    Returns:
        the flattened data

    >>> data = np.arange(18).reshape((2, 3, 3))
    >>> flatten(data).shape
    (2, 9)
    """
    return data.reshape(data.shape[0], -1)


@curry
def return_slice(x_data, cutoff):
    """
    returns region of interest around the center voxel upto the cutoff length
    """
    print(x_data.shape[1:])
    sliced = np.asarray(x_data.shape[1:]).astype(int) // 2
    print(sliced)
    if any(x < cutoff for x in sliced):
        raise NameError("Cut off is too large for the input array")
    print(type(sliced))
    make_slice = lambda i: slice(sliced[i] - cutoff, sliced[i] + cutoff + 1)

    if x_data.ndim == 3:
        return x_data[:,make_slice(0), make_slice(1)]
    if x_data.ndim == 4:
        return x_data[:,make_slice(0), make_slice(1), make_slice(2)]
    return Exception("Data should be either 2D or 3D")


@curry
def two_point_stats(boundary="periodic", cutoff=None, args0=None, args1=None):
    """
    Wrapper function that returns auto or crosscorrelations for
    input fields by calling appropriate modules.
    args:
        boundary : "periodic" or "nonperiodic"
        corrtype : "auto" or "cross"
        cutoff   :  cutoff radius of interest for the 2PtStatistics field
        args0    : 2D or 3D primary field of interest
        args1    : 2D or 3D field of interest which needs to be cross-correlated
                   with args1
    """

    ndim = args0.ndim
    # size = args0.size
    x_data = args0
    y_data = args1
    if cutoff is None:
        cutoff = args0.shape[0] // 2
    # Make sure this is working
    if boundary == "periodic":
        padder = lambda x: x
    elif boundary == "nonperiodic":
        padder = lambda x: np.pad(
            x, [(cutoff, cutoff)] * ndim, mode="constant", constant_values=0
        )
        x_data = padder(x_data)
        y_data = padder(y_data)
        ## This is a lot of redundant work
    return return_slice((corr_master(x_data, y_data) / x_data[0].size), cutoff)


class TwoPointcorrelation(BaseEstimator, TransformerMixin):
    """Reshape data ready for the LocalizationRegressor

    Sklearn likes flat image data, but MKS expects shaped data. This
    class transforms the shape of flat data into shaped image data for
    MKS.

    Attributes:
    Add test
    """

    def __init__(self, boundary="periodic", cutoff=None, correlations=None):
        """Instantiate a TwoPointcorrelation

        Args:
            boundary : "periodic" or "nonperiodic"
            corrtype : "auto" or "cross"
            cutoff   :  cutoff radius of interest for the 2PtStatistics field
            correlations: patial correlations to compute
        """
        self.boundary = boundary
        self.cutoff = cutoff
        self.xdata = correlations[0]
        self.ydata = correlations[1]

    def transform(self, x_input=None):
        """Transform the X data

            Args:
                x_data: the data to be transformed
        """
        x_data = x_input[:, :, :, self.xdata]
        y_data = x_input[:, :, :, self.ydata]

        if isinstance(x_data, np.ndarray):

            chunks = x_data.shape
            x_data = da.from_array(x_data, chunks=chunks)
        if isinstance(y_data, np.ndarray):
            chunks = y_data.shape
            y_data = da.from_array(y_data, chunks=chunks)

        return two_point_stats(
            boundary=self.boundary, cutoff=self.cutoff, args0=x_data, args1=y_data
        ).compute()

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

    def __init__(self):
        """Instantiate a FlattenTransformer

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
