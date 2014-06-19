# Installation

To get up and running with PyMKS, clone the repository

    $ git clone https://github.com/wd15/pymks.git

and install with 

    $ cd pymks
    $ python setup.py install
    
Check that you can import it and run the tests.

    $ python -c "import pymks; pymks.test()"

In all likelihood there will be some errors unless the packages below
are installed.

## Additional Requirements

PyMKS will not work without the following packages.

    $ pip install numpy
    $ pip install scipy
    $ pip install scikit-learn

## [SfePy][sfepy]

PyMKS can be used without [SfePy][sfepy]. However, a number of the
tests depend on [SfePy][sfepy] to run and it is also required to run
the elasticity examples.

To install [SfePy][sfepy], first clone with

    $ git clone git://github.com/sfepy/sfepy.git

and then install with

    $ cd sfepy
    $ python setup.py install

See the
[SfePy installation instructions](http://sfepy.org/doc-devel/installation.html)
for more details.

## [FFTW][fftw]

[FFTW][fftw] is not necessary to use PyMKS, but it does give improved
performance. The main requirement for `pyfftw` is FFTW3, which can be
installed via,

    $ sudo apt-get install libfftw3-dev
    
on Debian/Ubuntu. Windows and Mac users should consult

https://github.com/hgomersall/pyFFTW#platform-specific-build-info

Once this is installed, do

    $ pip install pyfftw
    
## Requirements

The [REQUIREMENTS.md](REQUIREMENTS.html) file has a complete list of
packages in the Python environment during development. However, most
of these are certainly not required for running PyMKS.

[sfepy]: http://sfepy.org 
[fftw]: http://www.fftw.org/




