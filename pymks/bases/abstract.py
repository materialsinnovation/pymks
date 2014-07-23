import numpy as np


class _AbstractMicrostructureBasis(object):
    def check(self, X, domain):
        dim = len(X.shape) - 1
        if dim == 0:
            raise RuntimeError("the shape of X is incorrect")
        if (np.min(X) < domain[0]) or (np.max(X) > domain[1]):
            raise RuntimeError("X must be within the specified range")

    def discretize(self, X):
        raise NotImplementedError

