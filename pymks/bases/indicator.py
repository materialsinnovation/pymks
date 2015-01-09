import numpy as np
from .abstract import _AbstractMicrostructureBasis


class _Indicator(_AbstractMicrostructureBasis):

    def _get_basis_slice(self, ijk, s0):
        """
        Helper method used to calibrate influence coefficients from in
        mks_regresison_model to account for redundancies from linearly
        dependent local states.
        """
        if np.all(np.array(ijk) == 0):
            s1 = s0
        else:
            s1 = (slice(-1),)
        return s1