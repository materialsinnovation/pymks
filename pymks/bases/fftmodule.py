"""Wrapper module for PyFFTW and Numpy's fft.

Provide a consistent interface for `rfftn`, `irfftn`, `fftn` and
`ifftn` between Numpy's fft and PyFFTW. PyFFTW takes extra arguments
that are not used by Numpy's fft module, for example, to call `rfftn`
with PyFFTW,

    rfft(np.ascontiguousarray(data),
         axes=axes,
         threads=threads,
         overwrite_input=True,
         planner_effort='FFTW_ESTIMATE',
         avoid_copy=True)

while using Numpy, it is just

    rfft(data,
         axes=axes,)

Furthermore, this module takes care of the logic for selecting whether
to use PyFFTW. If `PYMKS_USE_FFTW` is set in the environment and set to
a positive value then PyFFTW will be used. If `PYMKS_USE_FFTW` is not
set, then the following section in `setup.cfg` is used,

[pymks]
use-fftw = true

If `PYMKS_USE_FFTW` is set to a negative value then Numpy's fft module
is used. If `PYMKS_USE_FFTW` is unavailable and `use-fftw` is set to
false or is unavailable, then Numpy's fft module is used.

"""
import configparser
import os

import numpy.fft as numpy_fft
import numpy as np
from pkg_resources import resource_string
from ..fmks.func import deprecate

@deprecate
def config_use_pyfftw():
    """Determine the value of `use-fftw` in setup.cfg.

    Returns:
      boolean value based on `use-fftw` in setup.cfg
    """
    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = IOError
    try:
        config_string = resource_string('pymks', '../setup.cfg').decode('utf-8')
    except FileNotFoundError:
        return False
    parser = configparser.ConfigParser()
    parser.read_string(config_string)
    if parser.has_option('pymks', 'use-fftw'):
        return parser.getboolean('pymks', 'use-fftw')
    else:
        return False

@deprecate
def env_use_pyfftw():
    """Determine the value of the `PYMKS_USE_FFTW` environment variable.

    Returns:
      (defined, value): `defined` is bool depending on whether
        `PYMKS_USE_FFTW` is defined, `value` is the bool value it's set
        to (positive if undetermined)
    """
    from distutils.util import strtobool
    env_var = 'PYMKS_USE_FFTW'
    defined = env_var in os.environ
    if defined:
        env_string = os.environ[env_var]
        try:
            value = strtobool(env_string)
        except ValueError:
            value = True
    else:
        value = False
    return defined, value


@deprecate
def import_pyfftw():
    """Isolate pyfftw import

    Returns:
      the pyfftw.builders module
    """
    import pyfftw.builders as pyfftw_fft
    return pyfftw_fft


@deprecate
def choose_fftmodule():
    """Select either PyFFTW or Numpy's FFT

    Logic is:
      - pyfftw when `PYMKS_USE_FFT` is defined and positive
      - pyfftw when `PYMKS_USE_FFT` is defined and undetermined
      - numpy when `PYMKS_USE_FFT` is defined and negative
      - pyfftw when `PYMKS_USE_FFT` is not defined and config has `use_fftw = true`
      - numpy otherwise

    Returns:
      either `pyfftw.builders` or `numpy.fft` depending on logic
    """
    env_defined, env_value = env_use_pyfftw()
    if env_defined:
        if env_value:
            return import_pyfftw()
        else:
            return numpy_fft
    elif config_use_pyfftw():
        return import_pyfftw()
    else:
        return numpy_fft

FFTMODULE = choose_fftmodule()

@deprecate
def arg_wrap(fft_func, **extra_args):
    """Decorator to add kwargs based on fft suite.

    Args:
      fft_func: the fft function to arg wrap
      extra_args: extra args to add to the PyFFTW call

    Returns:
      the wrapped function
    """

    using_fftw = FFTMODULE.__name__.split('.')[0] == 'pyfftw'
    def wrapper(data, axes=None, **kwargs):
        """Wrapper function for Numpy's fft and PyFFTW

        Args:
          data: data to transform
          fftmodule: either numpy.fft or pyfftw.builders
          threads: the threads argument
          extra_args: extra args to add

        Returns:
          the transformed data
        """
        if using_fftw:
            kwargs.update(dict(planner_effort='FFTW_ESTIMATE',
                               avoid_copy=False))
            kwargs.update(extra_args)
            return fft_func(np.ascontiguousarray(data), axes=axes, **kwargs)()
        else:
            kwargs.pop('threads', None)
            return fft_func(data, axes=axes, **kwargs)
    return wrapper

@deprecate
def arg_wrap_overwrite(fft_func):
    """Add the overwrite_input=True argument to arg_wrap

    Args:
      fft_func: the fft function to arg wrap

    Returns:
      the wrapped function
    """
    return arg_wrap(fft_func, overwrite_input=True)


@arg_wrap_overwrite
def rfftn(data, axes=None, **kwargs):
    """Real Fourier transform wrapper

    Args:
      X: microstructure
      axes: the axes over which to perform the transform

    Returns:
      the transformation
    """
    return FFTMODULE.rfftn(data, axes=axes, **kwargs)

@arg_wrap
def irfftn(data, axes=None, axes_shape=None, **kwargs):
    """Inverse real Fourier transform wrapper

    Args:
      data: data to transform
      axes: the axes over which to perform the transform
      axes_shape: the axes shape

    Returns:
      the transformation
    """
    return FFTMODULE.irfftn(data, axes=axes, s=axes_shape, **kwargs)

@arg_wrap_overwrite
def fftn(data, axes=None, **kwargs):
    """Fourier transform wrapper

    Args:
      data: microstructure
      axes: the axes over which to perform the transform

    Returns:
      the transformation
    """
    return FFTMODULE.fftn(data, axes=axes, **kwargs)

@arg_wrap_overwrite
def ifftn(data, axes=None, **kwargs):
    """Inverse Fourier transform wrapper

    Args:
      data: microstructure
      axes: the axes over which to perform the transform

    Returns:
      the transformation
    """
    return FFTMODULE.ifftn(data, axes=axes, **kwargs)
