# Installation

The workshop will have workstations available in the CTCMS with a
correctly configured scientific Python environment already
installed. However, it is often preferable to use one's own laptop,
with a favored software and development environment. I much prefer
this. If you have experience with scientific Python, you probably
already have a satisfactory Python environment on your laptop, which
won't need many alterations. Just skip over the
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
    
and check that you are actually using Anaconda's Python. Once Anaconda
is installed, install the following,

    $ conda install pyfftw
    $ conda install ez_setup
    $ conda install fipy
    $ conda install scikit-learn
    $ conda install line_profiler
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
    $ pip install pyfftw
    $ pip install ez_setup
    $ pip install fipy
    $ pip install scikit-learn
    $ pip install line_profiler
    $ pip install numexpr
    
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
displayed. Click on the
[`01 - Python Intro`](1) notebook.  Run the cells under the "Test your
installation" heading near the top of the notebook. If this all works,
you are set, if not,
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

[1]: "http://nbviewer.ipython.org/github/wd15/pymks/blob/master/notebooks/01 - Python Intro"




