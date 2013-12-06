import fipy as fp
from fipy.solvers.scipy.linearLUSolver import LinearLUSolver
import numpy as np

class FiPyCHModel(object):
    def __init__(self, dx=0.25, dy=None, dt=1e-3):
        self.dx = dx
        if dy is None:
            self.dy = dx
        else:
            self.dy = dy
        self.dt = dt

    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        S, nx, ny = X.shape
        y = np.zeros(X.shape, dtype='d')
        mesh = fp.PeriodicGrid2D(nx=nx, ny=ny, dx=self.dx, dy=self.dy)
        phi = fp.CellVariable(mesh=mesh)
        PHI = phi.arithmeticFaceValue
        D = a = epsilon = 1
        eq = (fp.TransientTerm()
              == fp.DiffusionTerm(coeff=D * a**2 * (1 - 6 * PHI * (1 - PHI)))
              - fp.DiffusionTerm(coeff=(D, epsilon**2)))

        for i, x in enumerate(X):
            phi[:] = x.flatten()
            eq.solve(phi, dt=self.dt, solver=LinearLUSolver())
            phi_ij = np.array(phi).reshape((nx, ny))
            y[i] = (phi_ij - x) / self.dt 

        return y

