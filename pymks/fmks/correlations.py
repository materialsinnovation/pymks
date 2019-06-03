"""MKS Correlation Module
For computing auto and cross corelations under assumption
of  periodic or  non-periodic boundary conditions using discreete fourier
transform.


Note that input microstrucure should be 4 dimensional array.
where X=[n_sample,x,y.n_basis]
"""
import numpy as np
from toolz.curried import pipe, curry
from .func import dafftshift, dafftn, daifftn, daconj
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator
import dask.array as da

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

    s = np.asarray(x_data.shape).astype(int) // 2

    if x_data.ndim == 2:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1)]
    elif x_data.ndim ==3:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1),
                      (s[2] - cutoff):(s[2] + cutoff+1)]


@curry
def Two_point_stats(boundary="periodic", corrtype="auto", cutoff=None, args0=None, args1=None):
    """
    Wrapper function that returns auto or crosscorrelations for
    input fields by calling appropriate modules.
    args:
        boundary : "periodic" or "nonperiodic"
        corrtype : "auto" or "cross"
        cutoff   :  cutoff radius of interest for the 2PtStatistics field
	args0    : 2D or 3D primary field of interest
	args1    : 2D or 3D field of interest which needs to be cross-correlated with args1
    """

    ndim = args0.ndim
    size = args0.size
    x=args0
    if cutoff is None:
        cutoff = args0.shape[0] // 2
    cropper = return_slice(cutoff=cutoff)

    if boundary is "periodic":
        padder = lambda x: x
        if corrtype is "auto":
            y = None
        elif corrtype is "cross":
            y = args1
    elif boundary is "nonperiodic":
        padder = lambda x: np.pad(x, [(cutoff, cutoff),] * ndim, mode="constant", constant_values=0)
        if corrtype is "auto":
            y = None
        elif corrtype is "cross":
            y = padder(args1)

    return corr_master(x, y) / x[0].size if corrtype is "cross" else corr_master(x, x) / x[0].size

class TwoPointcorrelation(BaseEstimator, TransformerMixin):
    """Reshape data ready for the LocalizationRegressor

    Sklearn likes flat image data, but MKS expects shaped data. This
    class transforms the shape of flat data into shaped image data for
    MKS.

    Attributes:
    Add test
    """

    def __init__(self,boundary="periodic", corrtype="auto", cutoff=None,correlations=[1,0]):
        """Instantiate a TwoPointcorrelation

        Args:
            boundary : "periodic" or "nonperiodic"
            corrtype : "auto" or "cross"
            cutoff   :  cutoff radius of interest for the 2PtStatistics field
            correlations: patial correlations to compute
        """
        self.boundary=boundary
        self.corrtype=corrtype
        self.cutoff=cutoff
        self.xdata=correlations[0]
        self.ydata=correlations[1]

    def transform(self,X=None,y=None):
# Add a code for pulling the information for the fit_correlations

        x_data=X[:,:,:,self.xdata]
        y_data=X[:,:,:,self.ydata]
        """Transform the X data

        Args:
            x_data: the data to be transformed
        """
        if type(x_data) is np.ndarray:

            chunks=x_data.shape
            x_data=da.from_array(x_data,chunks=chunks)
        if type(y_data) is np.ndarray:
            chunks=y_data.shape
            y_data=da.from_array(y_data,chunks=chunks)

        return Two_point_stats(boundary=self.boundary, corrtype=self.corrtype, cutoff=self.cutoff, args0=x_data, args1=y_data).compute()

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
    def transform(self, x_data):
        """Transform the X data

        Args:
            x_data: the data to be transformed
        """
        return flatten(x_data)

    def fit(self, *_):
        """Only necessary to make pipelines work
        """
        return self
