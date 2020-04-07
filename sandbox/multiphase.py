"""Generate a random multiphase microstructure using a gaussian blur filter.

"""

import numpy as np
import dask.array as da
from toolz.curried import curry, pipe, compose
from scipy.ndimage.fourier import fourier_gaussian
from pymks.fmks.func import fftn, ifftn, fftshift, ifftshift, fmap

conj = curry(np.conjugate)
fabs = curry(np.absolute)

@curry
def _imfilter(x_data, f_data):
    """
    to convolve f_data over x_data
    """
    return pipe(f_data,
                ifftshift,
                fftn,
                lambda x: conj(x)*fftn(x_data),
                ifftn,
                fabs)


@curry
def _zero_pad(arr, shape):
    if len(shape) != len(arr.shape):
        raise RuntimeError("length of shape is incorrect")

    if not np.all(shape >= arr.shape):
        raise RuntimeError("resize shape is too small")

    return pipe(
        np.array(shape) - np.array(arr.shape),
        lambda x: np.concatenate(
            ((x - (x // 2))[..., None], (x // 2)[..., None]), axis=1
        ),
        fmap(tuple),
        tuple,
        lambda x: np.pad(arr, x, "constant", constant_values=0),
    )

@curry
def _gaussian_blur_filter(grain_size, domain_size):
    return pipe(grain_size, 
                lambda x: fourier_gaussian(np.ones(x), np.ones(len(x))), 
                fftshift,
                _zero_pad(shape=domain_size))


@curry
def _segmentation_values(x_data, n_samples, volume_fraction, n_phases):
    if volume_fraction is None:
        cumsum = pipe(n_phases, 
                      lambda x: [1/x] * x, 
                      lambda x: np.cumsum(x)[:-1])
    elif len(volume_fraction) == n_phases:
        cumsum = np.cumsum(volume_fraction)[:-1]

    return pipe(x_data, 
                lambda x: np.reshape(x, (n_samples,-1)),
                lambda x: np.quantile(x, q=cumsum, axis=1).T,
                lambda x: np.reshape(x, [n_samples,]+[1]*(x_data.ndim-1)+[n_phases-1,]))

@curry
def _npgenerate(n_phases = 5, 
                shape=(5, 101, 101), 
                grain_size= (25, 50),  
                volume_fraction=None, 
                seed=10):

    np.random.seed(seed)
    seg_values = _segmentation_values(n_samples=shape[0], 
                                         volume_fraction=volume_fraction, 
                                         n_phases=n_phases)
    np.random.seed(seed)
    return pipe(shape,
                lambda x: np.random.random(x),
                _imfilter(f_data=_gaussian_blur_filter(grain_size, shape[1:])),
                lambda x: x[...,None] > seg_values(x),
                lambda x: np.sum(x, axis=-1))

@curry
def generate(n_phases = 5, 
             shape=(5, 101, 101), 
             grain_size= (25, 50),  
             volume_fraction=None, 
             seed=10, 
             chunks=()):
    
    return da.from_array(_npgenerate(n_phases, shape, grain_size, 
                                     volume_fraction, seed), 
                         chunks=(chunks or (-1,))+shape)