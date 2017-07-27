# Installation

PyMKS is a pure Python package and is fully tested in both Python 2
and 3 and on Windows, Mac OSX and Linux.

## Conda

[Conda][conda] is the easiest way to install PyMKS. To install use

    $ conda install -c conda-forge pymks

This will install all the requirements necessary to use PyMKS.

    $ python -c "import pymks; pymks.test()"

See,
[https://www.continuum.io/downloads](https://www.continuum.io/downloads)
for more details on installing and using Conda.

## Pip

Use,

    $ pip install pymks

to install from [PyPI](https://pypi.org/). Further requirements listed
in the [requirements][requirements] file are necessary to use PyMKS
when installed using Pip.

## Scipy Stack

Both [Scipy](http://www.scipy.org/) and [Numpy][numpy] as well as
[Scikit-learn](http://scikit-learn.org) are required. See the
[requirements][requirements] for a full listing of PyMKS dependencies.

## Testing

To test use,

    $ python -c "import pymks; pymks.test()"

to run all the tests with an installed version of PyMKS.

## Examples

To use the interactive examples from the `notebooks/` directory,
Jupyter and Matplotlib are both required.

## [SfePy][sfepy]

PyMKS can be used without [SfePy][sfepy], but many of the tests and
examples require [SfePy][sfepy] to generate the sample data so it is a
good idea to install it. [Sfepy][Sfepy] will install automatically
with a Conda installation of PyMKS.

To install [SfePy][sfepy] manually, use

    $ conda -c conda-forge install sfepy

## [PyFFTW][pyfftw]


PyMKS can use [PyFFTW][pyfftw] to compute FFTs instead of
[Numpy][numpy]. As long as [Numpy][numpy] is not using
[Intel MKL][MKL], [PyFFTW][pyfftw] should improve the performance of
PyMKS. To use [PyFFTW][pyfftw], either set the environment variable

    $ export PYMKS_USE_FFTW=1

or set

    [pymks]
    use-fftw = true

in `setup.cfg` before installation.

To install [PyFFTW][pyfftw] use pip

    $ pip install pyfftw

or the Conda-Forge Conda channel,

    $ conda install -c conda-forge pyfftw

See the
[PyFFTW installation instructions](https://github.com/hgomersall/pyFFTW#installation)
for more details.

# Installation Issues

Please send questions and issues about installation of PyMKS to the
[pymks-general@googlegroups.com](mailto:pymks-general@googlegroups.com)
list.

[sfepy]: http://sfepy.org
[numpy]: http://www.scipy.org/
[MKL]: https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl
[pyfftw]: http://hgomersall.github.io/pyFFTW/
[conda]: http://continuum.io/downloads
[requirements]: https://raw.githubusercontent.com/materialsinnovation/pymks/master/requirements.txt