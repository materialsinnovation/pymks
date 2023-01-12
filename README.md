# PyMKS Overview

<a href="https://travis-ci.org/materialsinnovation/pymks" target="_blank">
<img src="https://api.travis-ci.org/materialsinnovation/pymks.svg"
alt="Travis CI">
</a>
<a href="https://github.com/materialsinnovation/pymks/blob/master/LICENSE.md">
<img src="https://img.shields.io/badge/license-mit-blue.svg" alt="License" height="18">
</a>
<a href="http://pymks.readthedocs.io/">
<img src="https://readthedocs.org/projects/pymks/badge/?version=latest" alt="Documentation Status" height="18">
</a>
<a href="https://pypi.python.org/pypi/pymks">
<img src="https://badge.fury.io/py/pymks.svg" alt="PyPI version" height="18">
</a>
<a href="https://circleci.com/gh/materialsinnovation/pymks">
<img src="https://circleci.com/gh/materialsinnovation/pymks.svg?style=shield" alt="Circle CI" height="18">
</a>

PyMKS is an open source, Pythonic implementation of the methodologies
developed under the aegis of Materials Knowledge System (MKS) to build
salient process-structure-property linkages for materials science
applications.  PyMKS provides for efficient tools for obtaining a
digital, uniform grid representation of a materials internal structure
in terms of its local states, and computing hierarchical descriptors
of the structure that can be used to build efficient machine learning
based mappings to the relevant response space.

The various materials data analytics workflows developed under the MKS
paradigm confirm to the data transformation pipeline architecture
typical to most Data Science workflows. The workflows can be boiled
down to a data preprocessing step, followed by a feature generation
step (fingerprinting), and a model construction step (including hyper
parameter optimization). PyMKS, written in a functional programming
style and supporting distributed computation (multi-core,
multi-threaded, cluster), provides modular functionalities to address
each of these data transformation steps, while maximally leveraging
the capabilities of the underlying computing environment.

PyMKS consists of tools to compute 2-point statistics, tools for both homogenization
and localization linkages, and tools for discretizing the microstructure. In addition,
PyMKS has modules for generating synthetic data sets using conventional numerical
simulations.

To learn about PyMKS start with the [PyMKS examples][EXAMPLES], especially the
[introductory example](notebooks/intro.ipynb). To learn more about the
methods consult the [technical overview](notebooks/tech_overview.ipynb)
for an introduction.

The two principle objects that PyMKS provides are the
[TwoPointCorrelation][TwoPointCorrelation]
transformer and the
[LocalizationRegressor][LocalizationRegressor]
which provide the homogenization and localization functionality. The
objects provided by PyMKS all work as either transformers or
regressors in a Scikit-Learn pipeline and use both Numpy and Dask
arrays for out-of-memory, distributed or parallel computations. The
out-of-memory computations are still in an experimental stage as of
version 0.4 and some issues still need to be resolved.

This effort has been supported with grants from NIST and the Vannevar Bush Fellowship to Professor Kalidindi at Georgia Tech.



## Feedback

Please submit questions and issues on the [GitHub issue
tracker](https://github.com/materialsinnovation/pymks/issues).

## Installation

### Conda

To install using [Conda][conda],

    $ conda install -c conda-forge pymks

To create a development environment clone this repository and run

    $ conda env create -f environment.yml
    $ conda activate pymks
    $ python setup.py develop

in the base directory.

### Pip

Install a minimal version of PyMKS with

    $ pip install pymks

This is enough to run the tests, but not the examples. Some optional
packages are not available via Pip. To create a development
environment clone this repository and run

    $ pip install .

in the base directory.

### Nix

Follow the [Nix installation
guide](https://nixos.org/nix/manual/#chap-quick-start) and then run

    $ export NIX_VERSION=21.05
    $ export PYMKS_VERSION=0.4.1
    $ nix-shell \
        -I nixpkgs=https://github.com/NixOS/nixpkgs/archive/${NIX_VERSION}.tar.gz \
        -I pymks=https://github.com/materialsinnovation/pymks/archive/tags/refs/v${PYMKS_VERSION}.tar.gz \
        -E 'with (import <nixpkgs> {}); mkShell { buildInputs = [ (python3Packages.callPackage <pymks> { graspi = null; }) ]; }'

to drop into a shell with PyMKS and all its requirements available. To
create a development environment with Nix clone this repository and
run

    $ nix-shell

in the base directory.

### Docker

PyMKS has a docker image avilable via
[docker.io](https://hub.docker.com/repository/docker/wd15/pymks). Assuming
that you have a working version of Docker, use

    $ docker pull docker.io/wd15/pymks
    $ docker run -i -t -p 8888:8888 wd15/fipy:latest
    # jupyter notebook --ip 0.0.0.0 --no-browser

The PyMKS example notebooks are available inside the image after
opening the Jupyter notebook from http://127.0.0.1:8888. See
[DOCKER.md](./DOCKER.md) for more details.

## Optional Packages

Packages that are optional when using PyMKS.

### Sfepy

[Sfepy](http://sfepy.org/doc-devel/index.html) is a python based
finite element solver. It's useful for generating data for PyMKS to
use for machine learning tasks. It's used in quite a few tests, but it
isn't strictly necessary to use PyMKS.  Sfepy will automatically
install when using Nix or Conda, but not when using Pip. See the
[Sfepy installation
instructions](http://sfepy.org/doc-devel/installation.html) to install
in your environment.

### GraSPI

GraSPI is a C++ library with a Python interface for creating materials
descriptors using graph theory. See the [API documentation][GRAPH]
for more details. Currently, only the Nix installation builds with
GraSPI by default To switch off GraSPI when using Nix use,

    $ nix-shell --arg withGraspi false

## Testing

To test a PyMKS installation use

    $ python -c "import pymks; pymks.test()"

## Citing

Please cite the following if you happen to use PyMKS for a
publication.

 - Brough, D.B., Wheeler, D. & Kalidindi, S.R. Materials Knowledge
   Systems in Python—a Data Science Framework for Accelerated
   Development of Hierarchical Materials. Integr Mater Manuf Innov 6,
   36–53 (2017). https://doi.org/10.1007/s40192-017-0089-0
   

[conda]: https://docs.conda.io/en/latest/
[EXAMPLES]: https://pymks.readthedocs.io/en/stable/EXAMPLES.html
[TwoPointCorrelation]: http://pymks.readthedocs.io/en/stable/API.html#pymks.TwoPointCorrelation
[LocalizationRegressor]: http://pymks.readthedocs.io/en/stable/API.html#pymks.LocalizationRegressor
[GRAPH]: http://pymks.readthedocs.io/en/stable/API.html#pymks.graph_descriptors

