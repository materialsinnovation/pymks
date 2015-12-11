from .primitive import PrimitiveBasis
from .legendre import LegendreBasis
from .fourier import FourierBasis

__all__ = ['PrimitiveBasis', 'LegendreBasis', 'FourierBasis']

DiscreteIndicatorBasis = PrimitiveBasis
ContinuousIndicatorBasis = PrimitiveBasis
