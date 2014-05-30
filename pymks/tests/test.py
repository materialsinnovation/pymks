from pymks import MKSRegressionModel
from pymks import ElasticFEModel
import numpy as np
from sklearn import metrics
mse = metrics.mean_squared_error

def test_ElasticFEModel():
    nx = 5
    ii = (nx - 1) / 2
    X = np.zeros((1, nx, nx, 2))
    X[..., 0] = 1.
    X[0, ii, ii, 0] = 10.
    X[..., 1] = 0.3
    
    model = ElasticFEModel(dx=1. / nx)
    strains = model.predict(X)
    solution = (1.518e-1, -1.672e-2, 0.)
    assert np.allclose(strains[0, ii, ii], solution, rtol=1e-3)

def get_delta_data(nx, ny):
    Lx = 1.
    dx = Lx / nx
    elastic_modulus = np.array((1., 1.1))
    poissons_ratio = np.array((0.3, 0.3))

    ii = (nx - 1) / 2
    jj = (ny - 1) / 2
    
    X = np.zeros((2, nx, ny), dtype=int)
    X[0, ii, jj] = 1
    X[1] = 1 - X[0]

    X_prop = np.concatenate((elastic_modulus[X][...,None], poissons_ratio[X][...,None]), axis=-1)
    elastic_model = ElasticFEModel(dx=dx)
    strains = elastic_model.predict(X_prop)

    return X, strains

def get_random_data(nx, ny):
    Lx = 1.
    dx = Lx / nx
    elastic_modulus = np.array((1., 1.1))
    poissons_ratio = np.array((0.3, 0.3))

    np.random.seed(10)
    X = np.random.randint(2, size=(1, nx, ny))

    X_prop = np.concatenate((elastic_modulus[X][...,None], poissons_ratio[X][...,None]), axis=-1)
    elastic_model = ElasticFEModel(dx=dx)
    strains = elastic_model.predict(X_prop)

    return X, strains

def test_MKSelastic_delta():
    nx, ny = 21, 21
    X, y_prop = get_delta_data(nx, ny)
    
    for y_test in np.rollaxis(y_prop, -1):
        model = MKSRegressionModel(Nbin=2)
        model.fit(X, y_test)
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y_test, rtol=1e-3, atol=1e-3)

def test_MKSelastic_random():
    nx, ny = 21, 21
    X_delta, strains_delta = get_delta_data(nx, ny)
    X_test, strains_test = get_random_data(nx, ny)

    def rollzip(x, y):
        return zip(np.rollaxis(x, -1), np.rollaxis(y, -1))
        
    for y_delta, y_test in rollzip(strains_delta, strains_test):
        model = MKSRegressionModel(Nbin=2)
        model.fit(X_delta, y_delta)
        y_pred = model.predict(X_test)
        a = y_pred[0].flatten()
        b = y_test[0].flatten()
        atol = 1e-3
        rtol = 1e-2
        allclose = -abs(a - b) + (atol + rtol * abs(b))
        print min(allclose)
        argmin = np.argmin(allclose)
        print a[argmin]
        print b[argmin]
        assert np.allclose(y_pred, y_test, rtol=1e-2, atol=1e-3)
        
if __name__ == '__main__':
    test_MKSelastic_random()
