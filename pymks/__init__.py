from mksRegressionModel import MKSRegressionModel
from fastmksRegressionModel import FastMKSRegressionModel
from fipyCHModel import FiPyCHModel
from tools import draw_microstructure_discretization
import pymks.mksRegressionModel
import pymks.fastmksRegressionModel
from tools import bin

def test():
    r"""
    Run all the doctests available.
    """
    import doctest
    doctest.testmod(pymks.mksRegressionModel)
    doctest.testmod(pymks.fastmksRegressionModel)
    doctest.testmod(pymks.tools)


def _getVersion():
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        version = get_distribution(__name__).version
    except DistributionNotFound:
        version = "unknown, try running `python setup.py egg_info`"
        
    return version
    
__version__ = _getVersion()           
