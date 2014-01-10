# Installation

The workshop will have workstations available in the CTCMS with a
correctly configured scientific Python environment already
installed. However, it is often preferable to use one's own laptop,
with your favored software and development environment. I much prefer
this. If you have experience with scientific Python, you probably
already have a satisfactory Python environment, which won't need many
alterations. Just skip over the
[Anaconda Python section](#anaconda-python) and install the packages
in the [Regular Python section](#regular-python).

## Anaconda Python 

If you have no experience with scientific Python, I would highly
recommend using the
[Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/),

https://store.continuum.io/cshop/anaconda/

I believe that it "just works" on all platforms and has mostly
everything included. Just follow the instructions for your
platform. To test whether Anaconda is working try

    $ python
    Python 2.7.6 |Anaconda 1.8.0 (64-bit)| (default, Nov 11 2013, 10:47:18) 
    ...
    >>> 1 + 1
    2
    >>> import numpy; numpy.__file__
    '/home/wd15/anaconda/lib/python2.7/site-packages/numpy/__init__.pyc'
    
and check that you are actually using Anaconda's Python. Then update Anaconda,

    $ conda update conda
    
Once Anaconda has updated, install the following.

    $ conda install ez_setup
    $ conda install fipy
    $ conda install scikit-learn
    $ conda install numexpr

## Regular Python

Install the following (if you are not using Anaconda). If you are a
seasoned Python user you probably have most of these, but you might
want to update them anyway, just use `pip install package --upgrade`.

    $ pip install ipython
    $ pip install numpy
    $ pip install scipy
    $ pip install matplotlib
    $ pip install scikit-learn
    $ pip install ez_setup
    $ pip install fipy
    $ pip install scikit-learn
    $ pip install numexpr

## FFTW

The main requirement for `pyfftw` is FFTW3, which can be installed
via,

    $ sudo apt-get install libfftw3-dev
    
on Debian/Ubuntu. Windows and Mac users should consult

https://github.com/hgomersall/pyFFTW#platform-specific-build-info

Once this is installed, do

    $ pip install pyfftw
    
If you have issues installing the FFTW3 requirement, don't worry. Most
of the notebooks do not require it and those that do are easily
modified to work with `numpy.fft`. We can also try and make this work
during the tutorial session.

## `line_profiler`

One additional package is `line_profiler`. Unfortunately, this has an
issue when installing with

    $ pip install line_profiler
    
If will throw an error. To install it, go to

https://pypi.python.org/pypi/line_profiler

and download and install the package by hand (rather than using
pip). On Windows, just run the `.exe` installer. On Mac and Linux,
download the `.tar.gz` and unpack it, change to the base directory and
run `python setup.py install`.

## Java Script Animator

This installation is not strictly necessary, but allows for nice
animations in the IPython Notebook. Clone it from Github,

    $ git clone https://github.com/jakevdp/JSAnimation
    
and then install with

    $ cd JSAnimation
    $ python setup.py install

If there are any issues, don't worry, this is only used in one place
near the end of the Python tutorial and there is an alternative way to
render the animation.

## Git

You'll need to use Git to clone the Github
[`pymks`](https://github.com/wd15/pymks) repository. There are Github
GUI style clients for both [Mac](http://mac.github.com/) and
[Windows](http://windows.github.com/). See,

https://help.github.com/articles/set-up-git
    
for help.

## Install the `pymks` module

Once you have Git installed, all you need to do is clone the
repository. At the command line this would be

    $ git clone https://github.com/wd15/pymks.git

The above should create a `pymks` directory with a working copy of the
repository. Next install `pymks` at the command line with,

    $ cd /PATH/TO/PYMKS/pymks
    $ python setup.py install
    
Check that you can import it and run the tests.

    $ python -c "import pymks; pymks.test()"

## IPython Notebooks

Check that you can launch and use the IPython notebooks

    $ cd /PATH/TO/PYMKS/pymks/notebooks
    $ ipython notebook

This should fire up your browser with the notebook dashboard
displayed. Click on the `01 - Python Intro` notebook.  Run the cells
under the "Test your installation" heading near the top of the
notebook. If this all works, you are set, if not,
[let me know](https://github.com/wd15/pymks/issues?state=open).

## Issues

If you have issues email me, daniel.wheeler@nist.gov, or better still
submit an issue on the
[issue tracker](https://github.com/wd15/pymks/issues?state=open) (it
emails me). The following are the issues I had when using Anaconda on
my Ubuntu laptop.

 - I had to change my backend in `.matplotlibrc` from `GTKAgg` to
   `TKAgg` when running `import matplotlib.pyplot as plt` from
   Anaconda. This issue should be apparent when running
   `pymks.test()`.
 
 - I had a blank screen in the browser when launching the IPython
   notebook. This may have been due to some dependency conflicts on my
   system as I have multiple versions of IPython. To overcome this, I
   cloned a development version of the notebook from the
   [IPython Github repo](https://github.com/ipython/ipython.git) and
   installed it. This seemed to work.

 - Andrew had the same blank screen issue in the CTCMS and he just
   switched to the Chrome browser and it worked.

