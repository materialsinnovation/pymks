import numpy as np
from .abstract import _AbstractMicrostructureBasis

    
class ContinuousIndicatorBasis(_AbstractMicrostructureBasis):
    def __init__(self, n_states=10):
        self.n_states = n_states

    def discretize(self, X):
        '''
        >>> n_states = 10
        >>> np.random.seed(4)
        >>> X = np.random.random((2, 5, 3, 2))
        >>> X_ = ContinuousIndicatorBasis(n_states=n_states).discretize(X)
        >>> H = np.linspace(0, 1, n_states)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)
        '''
        self.check(X, [0, 1])
        H = np.linspace(0, 1, self.n_states)
        return np.maximum(1 - (abs(X[..., None] - H)) / (H[1] - H[0]), 0)
        
