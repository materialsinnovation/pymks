import numpy as np
from .abstract import _AbstractMicrostructureBasis


class DiscreteIndicatorBasis(_AbstractMicrostructureBasis):
    r"""
    Digitize a discretly labeled microstructure such that,

    .. math::

       m_h \left[i\right] = \delta_{hm\left[i\right] }

    where :math:`0 \le h \le n-1` and :math:`0 \le m \le n-1` where
    :math:`n` is the number of states.  The following test is for a 2D
    (3x3) microstructre with three different phases labeled 0, 1 and
    2. The shape of `X` is `(n_samples, Nx, Ny)` where n_samples is 1
    in this case (`Nx=3` and `Ny=3`).

    >>> basis = DiscreteIndicatorBasis(n_states=3)
    >>> X = np.array([[[1, 1, 0],
    ...                [1, 0 ,2],
    ...                [0, 1, 0]]])
    >>> assert(X.shape == (1, 3, 3))

    The `discretize` method discretizes the labeled phases into three
    different local states. This results in an array of shape
    `(n_samples, Nx, Ny, n_states)`, where `n_states=3` in this case.
    For example, if a cell has a label of 2, its local state will be
    `[0, 0, 1]`. The local state can only have values of 0 or 1.

    >>> X_bin = np.array([[[[0, 1, 0],
    ...                     [0, 1, 0],
    ...                     [1, 0, 0]],
    ...                    [[0, 1, 0],
    ...                     [1, 0, 0],
    ...                     [0, 0, 1]],
    ...                    [[1, 0, 0],
    ...                     [0, 1, 0],
    ...                     [1, 0, 0]]]])
    >>> assert(np.allclose(X_bin, basis.discretize(X)))

    Check that the basis works when all the states are not specified
    in X.

    >>> basis = DiscreteIndicatorBasis(n_states=3)
    >>> X = np.array([1, 1, 0])
    >>> X_bin = np.array([[0, 1, 0],
    ...                   [0, 1, 0],
    ...                   [1, 0, 0]])
    >>> assert(np.allclose(X_bin, basis.discretize(X)))



    """
    def __init__(self, n_states=2, domain=None):
        super(DiscreteIndicatorBasis, self).__init__(n_states,
                                                     [0, n_states - 1])

    def discretize(self, X):
        '''
        Discretize `X`.

        Args:
          X: Integer valued field
        Returns:
          Integer valued field, either 0 or 1
        '''
        if not issubclass(X.dtype.type, np.integer):
            raise RuntimeError("X must be an integer array")
        self.check(X)

        Xbin = np.zeros(X.shape + (self.n_states,), dtype=float)
        mask = tuple(np.indices(X.shape)) + (X,)
        Xbin[mask] = 1.
        return Xbin
