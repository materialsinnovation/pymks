Materials Knowledge System Repository
=====================================

Authors
-------

-  `Daniel Wheeler <http://wd15.github.io/about.html>`__
-  `David Brough <https://github.com/davidbrough1>`__
-  `Tony
   Fast <http://mined.gatech.edu/the-ga-tech-mined-research-group/ga-tech-mined-research-group-tony-fast>`__
-  `Surya Kalidindi <http://www.me.gatech.edu/faculty/kalidindi>`__
-  `Andrew Reid <>`__

Overview
--------

The goal of this project is to provide a Python repository code base for
the Materials Knowledge System (MKS) as well as a set of example IPython
notebooks. The MKS is a a method of using spatial statistics to improve
the efficiency and efficacy of multiscale simulations and experimental
characterization involving materials microstructure. The techniques
outlined in the repository could be used in a range of applications,
however, the authors' primary interest is materials science
applications.

MKS Basics
----------

The details of the MKS are outlined in the notebooks. The main idea of
the MKS is to calculate responses from microstructures using expensive
calculations. The microstructures and responses are then linked with a
linear relationship via a set of influence coefficients, which can then
be used to make very fast calculations. See `the reference
section <#references>`__ for further reading.

References
----------

-  *Computationally-Efficient Fully-Coupled Multi-Scale Modeling of
   Materials Phenomena Using Calibrated Localization Linkages*, S. R.
   Kalidindi; ISRN Materials Science, vol. 2012, Article ID 305692,
   2012,
   `doi:10.5402/2012/305692 <http://dx.doi.org/10.5402/2012/305692>`__.

-  *Formulation and Calibration of Higher-Order Elastic Localization
   Relationships Using the MKS Approach*, Tony Fast and S. R. Kalidindi;
   Acta Materialia, vol. 59 (11), pp. 4595-4605, 2011,
   `doi:10.1016/j.actamat.2011.04.005 <http://dx.doi.org/10.1016/j.actamat.2011.04.005>`__

-  *Developing higher-order materials knowledge systems*, T. N. Fast;
   Thesis (PhD, Materials engineering)--Drexel University, 2011,
   `doi:1860/4057 <http://dx.doi.org/1860/4057>`__.

License
-------

The repository is licensed with the FreeBSD License, see
`LICENSE.txt <LICENSE.txt>`__.

Installation
------------

See `INSTALLATION.md <INSTALLATION.md>`__.

Requirements
------------

The `REQUIREMENTS.txt <REQUIREMENTS.txt>`__ file has a complete list of
packages in the Python environment during development. The most
important of these are listed. The version numbers are mostly not
important within reason, but if you have problems the version numbers
may help.

-  FiPy dev version ``6e897df40012``
-  IPython dev version ``b31eb2f2d951``
-  Matplotlib 1.2.1
-  Numpy 1.7.1
-  Scikit-learn 0.13.1
-  Scipy 0.13.0
-  pyFFTW 0.9.2
-  ez-setup 0.9
-  line-profiler 1.0b3
-  numexpr 2.2.2

Citing
------

See `CITATION.md <CITATION.md>`__.

Viewing the Notebooks
---------------------

The Notebooks can be viewed at
`nbviewer.ipython.org <http://nbviewer.ipython.org/github/wd15/pymks/tree/master/notebooks/>`__
and are automatically updated as changes are pushed to the repository.

Testing
-------

To run the tests use

::

    $ python -c "import pymks; pymks.test()"

Note that the above does not test the notebooks, only the modules under
``pymks/``. Integrating the notebooks with the test harness is an
ongoing concern.
