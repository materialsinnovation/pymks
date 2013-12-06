from mksRegressionModel import MKSRegressionModel
from mksRegressionModelSlow import MKSRegressionModelSlow
from fipyCHModel import FiPyCHModel

import pymks.mksRegressionModel

def test():
    r"""
    Run all the doctests available.
    """
    import doctest
    doctest.testmod(pymks.mksRegressionModel)


def _getVersion():
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        version = get_distribution(__name__).version
    except DistributionNotFound:
        version = "unknown, try running `python setup.py egg_info`"
        
    return version
    
__version__ = _getVersion()           
