Installation
============

Use pip,

::

    $ pip install pymks

and then run the tests.

::

    $ python -c "import pymks; pymks.test()"

Scipy Stack
-----------

The packages `Nosetests <https://nose.readthedocs.org/en/latest/>`__,
`Scipy <http://www.scipy.org/>`__, `Numpy <http://www.scipy.org/>`__,
`Scikit-learn <http://scikit-learn.org>`__ are all required.

Examples
--------

To use the interactive examples from the ``notebooks/`` directory,
IPython and Matplotlib are both required.

`SfePy <http://sfepy.org>`__
----------------------------

PyMKS can be used without `SfePy <http://sfepy.org>`__, but many of the
tests and examples require `SfePy <http://sfepy.org>`__ to generate the
sample data so it is a good idea to install it.

To install `SfePy <http://sfepy.org>`__, first clone with

::

    $ git clone git://github.com/sfepy/sfepy.git

and then install with

::

    $ cd sfepy
    $ python setup.py install

See the `SfePy installation
instructions <http://sfepy.org/doc-devel/installation.html>`__ for more
details.

Requirements
------------

The `REQUIREMENTS.md <REQUIREMENTS.html>`__ file has a complete list of
packages in the Python environment during development. However, most of
these are certainly not required for running PyMKS.
