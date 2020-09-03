<p align="center">
<img src="https://raw.githubusercontent.com/materialsinnovation/pymks/master/doc/pymks_logo.ico"
     height="60"
     alt="PyMKS logo"
     class="inline">
</p>

<h1> <p align="center"><sup><strong>
PyMKS &ndash; The Materials Knowledge System in Python
</strong></sup></p>
</h1>

<a href="https://travis-ci.org/materialsinnovation/pymks" target="_blank">
<img src="https://api.travis-ci.org/materialsinnovation/pymks.svg"
alt="Travis CI">
</a>
<a href="https://github.com/materialsinnovation/pymks/blob/master/LICENSE.md">
<img src="https://img.shields.io/badge/license-mit-blue.svg" alt="License" height="18">
</a>
<a href="http://pymks.readthedocs.io/en/latest/?badge=latest">
<img src="https://readthedocs.org/projects/pymks/badge/?version=latest" alt="Documentation Status" height="18">
</a>
<a href="https://pypi.python.org/pypi/pymks">
<img src="https://badge.fury.io/py/pymks.svg" alt="PyPI version" height="18">
</a>
<a href="https://circleci.com/gh/materialsinnovation/pymks">
<img src="https://circleci.com/gh/materialsinnovation/pymks.svg?style=shield" alt="Circle CI" height="18">
</a>


PyMKS is an open source, pythonic implementation of the methodologies
developed under the aegis of Materials Knowledge System (MKS) to build
salient process-structure-property linkages for materials science applications.
PyMKS provides for efficient tools for obtaining a digital, uniform grid representation
of a materials internal structure in terms of its local states, and computing hierarchical
descriptors of the structure that can be used to build efficient machine
learning based mappings to the relevant response space.


The various materials data analytics workflows developed under the MKS paradigm confirm to
the data transformation pipeline architecture typical to most Data Science workflows. The workflows
can be boiled down to a data preprocessing step, followed by a feature generation step (fingerprinting),
and a model construction step (including hyper parameter optimization). PyMKS, written in a functional
programming style and supporting distributed computation (multi-core, multi-threaded, cluster), provides
modular functionalities to address each of these data transformation steps, while maximally leveraging
the capabilities of the underlying computing environment.


PyMKS consists of tools to compute 2-point statistics, tools for both homogenization
and localization linkages, and tools for discretizing the microstructure. In addition,
PyMKS has modules for generating synthetic data sets using conventional numerical
simulations.

To learn about PyMKS start with the [PyMKS examples](./index.ipynb),
especially the [introductory example](notebooks/intro.ipynb).
To learn more about the methods consult the
[technical overview](http://pymks.org/en/latest/rst/notebooks/tech_overview.html)
for an introduction.


The two principle objects that PyMKS provides are the
`TwoPointCorrelation` transformer and the `LocalizationRegressor`
which provide the homogenization and localization functionality. The
objects provided by PyMKS all work as either transformers or
regressors in a Scikit-Learn pipeline and use both Numpy and Dask
arrays for out-of-memory, distributed or parallel computations. The
out-of-memory computations are still in an experimental stage as of
version 0.4 and some issues still need to be resolved.

## Feedback

Please submit questions and issues on the [GitHub issue
tracker](https://github.com/materialsinnovation/pymks/issues).

## Installation

### Conda

To install using [Conda][conda],

    $ conda install -c conda-forge pymks

or to create a development environment use,

    $ conda env create -f environment.yml
    $ conda activate pymks
    $ python setup.py develop

### Pip

Install a minimal version of PyMKS with

    $ pip install pymks

This is enough to run the tests, but not the examples. Some optional
packages are not available via Pip. To create a development
environment use,

    $ pip install .

### Nix

Follow the [Nix installation
guild](https://nixos.org/nix/manual/#chap-quick-start) and then run

    $ nix-shell

to drop into a shell with PyMKS and all its requirements available.

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
