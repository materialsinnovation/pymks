import os
import pytest
from .mks_localization_model import MKSLocalizationModel
from .bases.primitive import PrimitiveBasis
from .bases.legendre import LegendreBasis
from .mks_homogenization_model import MKSHomogenizationModel
from .mks_structure_analysis import MKSStructureAnalysis
MKSRegressionModel = MKSLocalizationModel
DiscreteIndicatorBasis = PrimitiveBasis
ContinuousIndicatorBasis = PrimitiveBasis


def test():
    """
    Run all the doctests available.
    """
    try:
        import sfepy
    except ImportError:
        print "==============================================================="
        print "Optional package 'sfepy' not found. Some tests will be skipped."
    path = os.path.split(__file__)[0]
    pytest.main(args=[path, '--doctest-module'])


def get_version():
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        version = get_distribution(__name__).version
    except DistributionNotFound:
        version = "unknown, try running `python setup.py egg_info`"

    return version

__version__ = get_version()

__all__ = ['__version__',
           'test',
           'MKSLocalizationModel',
           'PrimitiveBasis',
           'LegendreBasis',
           'MKSHomogenizationModel',
           'MKSStructureAnalysis']
