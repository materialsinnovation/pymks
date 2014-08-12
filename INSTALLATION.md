# Installation

Use pip,

    $ pip install pymks

Check that you can import it and run the tests.

    $ python -c "import pymks; pymks.test()"

## Scipy Stack

The packages Nosetests, Scipy, Numpy, Scikit-learn and [Sfepy](sfepy)
are all required to run the tests.

## Examples

To use the interactive examples from the `notebooks/` directory,
IPython and Matplotlib are required.

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

## Requirements

The [REQUIREMENTS.md](REQUIREMENTS.html) file has a complete list of
packages in the Python environment during development. However, most
of these are certainly not required for running PyMKS.

[sfepy]: http://sfepy.org 




