import numpy as np
from .abstract import _AbstractMicrostructureBasis    


class DiscreteIndicatorBasis(_AbstractMicrostructureBasis):
    """
    Digitize a discretly labeled microstructre.

    The following test is for a 2D (3x3) microstructre with three
    different phases labeled 0, 1 and 2. The shape of `X` is
    `(n_samples, Nx, Ny)` where n_samples is 1 in this case (`Nx=3`
    and `Ny=3`).
    
    >>> basis = DiscreteIndicatorBasis()
    >>> X = np.array([[[1, 1, 0],
    ...                [1, 0 ,2],
    ...                [0, 1, 0]]])
    >>> print X.shape
    (1, 3, 3)

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

    
    """
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
        self.check(X, [0, np.max(X)])

        n_states = np.max(X) + 1
        Xbin = np.zeros(X.shape + (n_states,), dtype=float)
        mask = tuple(np.indices(X.shape)) + (X,)
        Xbin[mask] = 1.
        return Xbin
