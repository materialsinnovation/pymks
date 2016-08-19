![PyMKS Logo](http://github.com/wd15/pymks/doc/pymks_logo.ico)

<p align="center"><sup><strong>
The Materials Knowledge System in Python.
</strong></sup></p>

<p align="center">
<a href="http://mybinder.org/repo/materialsinnovation/pymks" target="_blank">
<img src="http://mybinder.org/badge.svg"
alt="Binder">
</a>
<a href="https://gitter.im/pymks/Lobby" target="_blank">
<img src="https://img.shields.io/gitter/room/gitterHQ/gitter.svg"
alt="Gitter Chat">
</a>
<a href="https://travis-ci.org/materialsinnovation/pymks" target="_blank">
<img src="https://api.travis-ci.org/materialsinnovation/pymks.svg"
alt="Travis CI">
</a>
<a href="https://pypi.python.org/pypi/pymks/0.3.1">
<img src="https://badge.fury.io/py/pymks.svg" alt="PyPI version" height="18">
</a>
<a href="LICENSE.md">
<img src="https://img.shields.io/badge/license-mit-blue.svg" alt="License" height="18">
</a>
</p>

### MKS

The Materials Knowledge Systems (MKS) is a novel data science approach
for solving multiscale materials science problems. It uses techniques
from physics, machine learning, regression analysis, signal processing,
and spatial statistics to create processing-structure-property
relationships. The MKS carries the potential to bridge multiple
length scales using localization and homogenization linkages, and
provides a data driven framework for solving inverse material design
problems.

See these references for further reading:

 - *Computationally-Efficient Fully-Coupled Multi-Scale Modeling of
   Materials Phenomena Using Calibrated Localization Linkages*,
   S. R. Kalidindi; ISRN Materials Science, vol. 2012, Article ID
   305692, 2012,
   [doi:10.5402/2012/305692](http://dx.doi.org/10.5402/2012/305692).

 - *Formulation and Calibration of Higher-Order Elastic Localization
   Relationships Using the MKS Approach*, Tony Fast and
   S. R. Kalidindi; Acta Materialia, vol. 59 (11), pp. 4595-4605,
   2011,
   [doi:10.1016/j.actamat.2011.04.005](http://dx.doi.org/10.1016/j.actamat.2011.04.005)

 - *Developing higher-order materials knowledge systems*, T. N. Fast;
   Thesis (PhD, Materials engineering)--Drexel University, 2011,
   [doi:1860/4057](http://dx.doi.org/1860/4057).

### PyMKS

The Materials Knowledge Materials in Python (PyMKS) framework is an
object-oriented set of tools and examples, written in Python, that
provide high-level access to the MKS framework for rapid creation and
analysis of structure-property-processing relationships. A short
introduction to how to use PyMKS is outlined below and example cases can
be found [in the examples section](EXAMPLES.html). Both code and
examples contributions are welcome.

### Documentations

Start with the [PyMKS examples](./index.ipynb), especially the
[introductory example](notebooks/intro.ipynb).

### Other

#### Mailing List

Please feel free to ask open-ended questions about PyMKS on the
<pymks-general@googlegroups.com> list.

#### Docker

The [Dockerfile](Dockerfile) is for Binder, but can be used
locally. See [ADMINISTRATA.md](ADMINISTRATA.md) for more details. The
official PyMKS instance is at https://hub.docker.com/r/wd15/pymks/.
