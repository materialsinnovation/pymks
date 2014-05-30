from pymks import MKSRegressionModel
from pymks import ElasticFEModel
import numpy as np
from sklearn import metrics
mse = metrics.mean_squared_error

def test_ElasticFEModel():
    X = np.zeros((1, 5, 5, 2))
    X[..., 0] = 10.
    X[0, 2, 2, 0] = 1.
    X[..., 1] = 0.3
    
    model = ElasticFEModel()
    y = model.predict(X)
    assert np.allclose(y[0, 2, 2, :], (4.1987e-1, 6.5947e-2, 0), rtol=1e-4)

def MKSelastic(nx, ny):
    elastic_modulus = np.array((1., 1.5))
    poissons_ratio = np.array((0.3, 0.3))

    Lx = 1.
    dx = Lx / nx
    ii = (nx - 1) / 2
    jj = (ny - 1) / 2
    
    X = np.zeros((2, nx, ny), dtype=int)
    X[0, ii, jj] = 1
    X[1] = 1 - X[0]

    X_prop = np.concatenate((elastic_modulus[X][...,None], poissons_ratio[X][...,None]), axis=-1)

    elastic_model = ElasticFEModel(dx=dx)
    y_prop = elastic_model.predict(X_prop)

    for y_test in np.rollaxis(y_prop, -1):
        model = MKSRegressionModel(Nbin=2)
        model.fit(X, y_test)
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y_test, rtol=1e-1, atol=1e-2)

def test_MKSelastic():
    MKSelastic(11, 11)
    MKSelastic(15, 11)
if __name__ == '__main__':
    test_MKSelastic()
