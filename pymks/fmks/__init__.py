"""fMKS - functional matierals knowledge system.

See https://github.com/wd15/fmks

fMKS is a functional version of [PyMKS](https://pymks.org) currently
under development. The purpose of the project is to prototype a
parallel implementation of MKS using functional programming in Python
primarily using the [Toolz](http://toolz.readthedocs.io) library.

"""

import os


def test():  # pragma: no cover
    r"""
    Run all the doctests available.
    """
    import pytest
    path = os.path.split(__file__)[0]
    pytest.main(args=[path,
                      '--doctest-modules',
                      '--ignore=setup.py ',
                      '-r s',
                      '--cov=fmks'])


def get_version() -> str:
    """Get the version of the code from egg_info.

    Returns:
      the package version number
    """
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        # pylint: disable=no-member
        version = get_distribution(__name__.split('.')[0]).version
    except DistributionNotFound:  # pragma: no cover
        version = "unknown, try running `python setup.py egg_info`"

    return version


__version__ = get_version()

__all__ = ['__version__',
           'test']
