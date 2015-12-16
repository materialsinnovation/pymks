from .primitive import PrimitiveBasis
from .legendre import LegendreBasis
from .fourier import FourierBasis
from .gsh import GSHBasis

__all__ = ['PrimitiveBasis', 'LegendreBasis', 'FourierBasis', 'GSHBasis']

DiscreteIndicatorBasis = PrimitiveBasis
ContinuousIndicatorBasis = PrimitiveBasis
