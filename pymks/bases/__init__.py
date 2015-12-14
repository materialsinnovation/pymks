from .primitive import PrimitiveBasis
from .legendre import LegendreBasis
from .fourier import FourierBasis
from .gshhexagonal import GSHBasisHexagonal

__all__ = ['PrimitiveBasis', 'LegendreBasis', 'FourierBasis', 'GSHBasisHexagonal']

DiscreteIndicatorBasis = PrimitiveBasis
ContinuousIndicatorBasis = PrimitiveBasis
