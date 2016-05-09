import os
import nose
from nose.tools import nottest
from .mks_localization_model import MKSLocalizationModel
from .bases.primitive import PrimitiveBasis
from .bases.legendre import LegendreBasis
from .mks_structure_analysis import MKSStructureAnalysis
from .mks_homogenization_model import MKSHomogenizationModel
MKSRegressionModel = MKSLocalizationModel
DiscreteIndicatorBasis = PrimitiveBasis
ContinuousIndicatorBasis = PrimitiveBasis


@nottest
def test():
    r"""
    Run all the doctests available.
    """
    path = os.path.split(__file__)[0]
    nose.main(argv=['-w', path, '--with-doctest'])


def get_version():
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        version = get_distribution(__name__).version
    except DistributionNotFound:
        version = "unknown, try running `python setup.py egg_info`"

    return version

def module_exists(module_name):
    """Check if module is installable.

    Args:
      module_name: the name of the module.

    Returns:
      True if exists and available, False otherwise.

    """
    import imp
    try:
        imp.find_module(module_name)
        return True
    except ImportError:
        return False

def skip_sfepy(func):
    from functools import wraps
    import unittest
    @wraps(func)
    @unittest.skipIf(not module_exists("sfepy"), "Sfepy not available")
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
    return wrapper

__version__ = get_version()

__all__ = ['__version__',
           'test',
           'MKSLocalizationModel',
           'PrimitiveBasis',
           'LegendreBasis',
           'MKSHomogenizationModel',
           'MKSStructureAnalysis']
