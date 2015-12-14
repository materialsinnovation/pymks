# Installation
The following steps outline the necessary requirements for a successful installation of PyMKS.

Use pip,

    $ pip install pymks

and then run the tests.

    $ python -c "import pymks; pymks.test()"

## Scipy Stack

The packages [Nosetests](https://nose.readthedocs.org/en/latest/),
[Scipy](http://www.scipy.org/), [Numpy][numpy], and
[Scikit-learn](http://scikit-learn.org) are all required.

## Examples

To use the interactive examples from the `notebooks/` directory,
IPython and Matplotlib are both required.

## [SfePy][sfepy]

PyMKS can be used without [SfePy][sfepy], but many of the tests and
examples require [SfePy][sfepy] to generate the sample data so it is a
good idea to install it.

To install [SfePy][sfepy], first clone with

    $ git clone git://github.com/sfepy/sfepy.git

and then install with

    $ cd sfepy
    $ python setup.py install

See the
[SfePy installation instructions](http://sfepy.org/doc-devel/installation.html)
for more details.

## [PyFFTW][pyfftw]

If installed, PyMKS will use [PyFFTW][pyfftw] to
computed FFTs instead of [Numpy][numpy]. As long as [Numpy][numpy] is
not using [Intel MKL][MKL], [PyFFTW][pyfftw] should improvement the
performance of PyMKS.

To install [PyFFTW][pyfftw] use pip

    $ pip install pyfftw

See the [PyFFTW installation instructions](https://github.com/hgomersall/pyFFTW#installation)
 for more details.

## Installation on Windows

We recommend you download and install the [Anaconda Python Distribution](http://continuum.io/downloads)
for Python 2.7 (x64) and then download and install PyMKS using the [windows installer](https://github.com/materialsinnovation/pymks/releases/download/version-0_2_1/PyMKS-x64-anaconda27.exe).

## Installation on Mac OS X

We recommend you download and install the [Anaconda Python Distibution](http://continuum.io/downloads)
for Python 2.7 (x64). Once Anaconda has been installed, follow the above procedures to install SfePy.
Finally, install PyMKS using `pip` as described above.

## Installation with Anaconda

The [Anaconda Python Distributionn](https://store.continuum.io/cshop/anaconda/)
contains all of the required packages outside of [SfePy][sfepy] and
works on multiple platforms. [Download][conda] and
[install](http://docs.continuum.io/anaconda/install.html) Anaconda, and
use your terminal or shell to install PyMKS using pip.

## Requirements

The [REQUIREMENTS.md](https://github.com/materialsinnovation/pymks/blob/master/REQUIREMENTS.md) file has a list of required
packages in a Python environment used to run tests and examples
for the current release of PyMKS.

#Installation Issues

Please send questions and issues about installation of PyMKS to the
[pymks-general@googlegroups.com](mailto:pymks-general@googlegroups.com)
list.

[sfepy]: http://sfepy.org
[numpy]: http://www.scipy.org/
[MKL]: https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl
[pyfftw]: http://hgomersall.github.io/pyFFTW/
[chris]: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn
[conda]: http://continuum.io/downloads
