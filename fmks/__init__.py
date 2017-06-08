import os


def test():
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

def get_version():
    """Get the version of the code from egg_info.

    Returns:
      the package version number
    """
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        version = get_distribution(__name__.split('.')[0]).version # pylint: disable=no-member
    except DistributionNotFound: # pragma: no cover
        version = "unknown, try running `python setup.py egg_info`"

    return version

__version__ = get_version()

__all__ = ['__version__',
           'test']
