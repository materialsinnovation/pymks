# Materials Knowledge System Tutorial

## Schedule

See https://github.com/wd15/pymks/wiki/Workshop-Schedule

## Overview

The goal of this project is to provide a tutorial for the Materials
Knowledge System (MKS) as a set of IPython notebooks. The MKS is a a
method of using spatial statistics to improve the efficiency and
efficacy of multiscale simulations and experimental characterization
involving materials microstructure. The techniques outlined in the
tutorials could be used in range of applications, however, the
authors' primary interest is materials science applications.


## MKS Basics

The details of the MKS are outlined in the notebooks. The main idea of
the MKS relates is to calculate responses from microstructures using
expensive calculations. The microstructures and responses are then
linked with a linear relationship via a set of influence coefficients,
which can then be used to make very fast calculations. See
[Tony Fast's Thesis](http://idea.library.drexel.edu/bitstream/1860/4057/1/Fast_AnthonyPhD.pdf)
for complete details.

## License

The repository is licensed with the FreeBSD License, see
[LICENSE.txt](LICENSE.txt).

## Installation

See [INSTALLATION.md](INSTALLATION.md).

## Requirements

The [REQUIREMENTS.txt](REQUIREMENTS.txt) file has a complete list of
packages in the Python environment during development. The most
important of these are listed. The version numbers are mostly not
important within reason, but if you have problems the version numbers
may help.

 * FiPy dev version `6e897df40012`
 * IPython dev version `b31eb2f2d951`
 * Matplotlib 1.2.1
 * Numpy 1.7.1
 * Scikit-learn 0.13.1
 * Scipy 0.13.0
 * pyFFTW 0.9.2
 * ez-setup 0.9
 * line-profiler 1.0b3
 * numexpr 2.2.2

## Citing the Notebooks

The plan is to either add a citable DOI for the entire project or for
each individual notebook via Figshare (or Authorea).

## Viewing the Notebooks

The Notebooks can be viewed at
[nbviewer.ipython.org](http://nbviewer.ipython.org/github/wd15/pymks/tree/master/notebooks/)
and are automatically updated as changes are pushed to the repository.

## Authors

 * [Daniel Wheeler](http://wd15.github.io/about.html)
 * [Tony Fast](http://mined.gatech.edu/the-ga-tech-mined-research-group/ga-tech-mined-research-group-tony-fast)

## Testing

To run the tests use

    $ python -c "import pymks; pymks.test()"

Note that the above does not test the notebooks, only the modules
under `pymks/`. Integrating the notebooks with the test harness is an
ongoing concern.
