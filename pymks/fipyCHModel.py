import fipy as fp
from fipy.solvers.scipy.linearLUSolver import LinearLUSolver
import numpy as np

class FiPyCHModel(object):
    def __init__(self, dx=0.005, dy=None, dt=1e-8, a=np.sqrt(200.), epsilon=0.1):
        self.dx = dx
        if dy is None:
            self.dy = dx
        else:
            self.dy = dy
        self.dt = dt
        self.a = a
        self.epsilon = epsilon

    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        S, nx, ny = X.shape
        y = np.zeros(X.shape, dtype='d')
        mesh = fp.PeriodicGrid2D(nx=nx, ny=ny, dx=self.dx, dy=self.dy)
        phi = fp.CellVariable(mesh=mesh, hasOld=True)
        PHI = phi.arithmeticFaceValue
        D = 1
        eq = (fp.TransientTerm()
              == fp.DiffusionTerm(coeff=D * self.a**2 * (1 - 6 * PHI * (1 - PHI)))
              - fp.DiffusionTerm(coeff=(D, self.epsilon**2)))

        for i, x in enumerate(X):
            phi[:] = x.flatten()
            phi.updateOld()
            self._solve(eq, phi, LinearLUSolver(), self.dt)
            #            eq.solve(phi, dt=self.dt, solver=LinearLUSolver())
            phi_ij = np.array(phi).reshape((nx, ny))
            y[i] = (phi_ij - x) / self.dt 

        return y

    def _solve(self, eq, phi, solver, dt):
        res = 1e+10
        
        for sweep in range(5):
            res = eq.sweep(phi, dt=dt, solver=LinearLUSolver())
            #    print 'res',res
            

