import numpy as np
from .abstract import _AbstractMicrostructureBasis    


class DiscreteIndicatorBasis(_AbstractMicrostructureBasis):
    def discretize(self, X):
        '''Create microstruture function for integer valued microstrutures
        >>> basis = DiscreteIndicatorBasis()
        >>> X = np.array([[1, 1, 0],
        ...               [1, 0 ,2],
        ...               [0, 1, 0]])

        >>> X_bin = np.array([[[0, 1, 0],
        ...                    [0, 1, 0],
        ...                    [1, 0, 0]],
        ...                   [[0, 1, 0],
        ...                    [1, 0, 0],
        ...                    [0, 0, 1]],
        ...                   [[1, 0, 0],
        ...                    [0, 1, 0],
        ...                    [1, 0, 0]]])
        >>> assert(np.allclose(X_bin, basis.discretize(X)))

        Args:
          X: Interger valued microstructure
        Returns:
          Interger valued microstructure function
        '''
        if not issubclass(X.dtype.type, np.integer):
            raise RuntimeError("X must be an integer array")
        self.check(X, [0, np.max(X)])

        n_states = np.max(X) + 1
        Xbin = np.zeros(X.shape + (n_states,), dtype=float)
        mask = tuple(np.indices(X.shape)) + (X,)
        Xbin[mask] = 1.
        return Xbin
