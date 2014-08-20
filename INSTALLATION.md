# Installation

Use pip,

    $ pip install pymks

and then run the tests.

    $ python -c "import pymks; pymks.test()"

## Scipy Stack

The packages [Nosetests](https://nose.readthedocs.org/en/latest/),
[Scipy](http://www.scipy.org/), [Numpy](http://www.scipy.org/),
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

## Requirements

The [REQUIREMENTS.md](REQUIREMENTS.html) file has a complete list of
packages in the Python environment during development. However, most
of these are certainly not required for running PyMKS.

[sfepy]: http://sfepy.org 




