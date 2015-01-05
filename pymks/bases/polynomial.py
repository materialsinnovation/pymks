from .abstract import _AbstractMicrostructureBasis


class _Polynomial(_AbstractMicrostructureBasis):

    def _get_basis_slice(self, ijk, s0):
        """
        Helper method used to calibrate influence coefficients from in
        mks_regresison_model to account for redundancies from linearly
        dependent local states.
        """
        return s0