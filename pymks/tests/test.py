from pymks import MKSRegressionModel
from pymks.datasets.elasticFESimulation import ElasticFESimulation
import numpy as np
from sklearn import metrics
mse = metrics.mean_squared_error
from pymks.datasets import make_elasticFEstrain_delta
from pymks.datasets import make_elasticFEstrain_random
from pymks.datasets.cahnHilliardSimulation import CahnHilliardSimulation
from pymks.datasets import make_cahnHilliard
from pymks.bases import DiscreteIndicatorBasis

def test_ElasticFESimulation_2D():
    nx = 5
    ii = (nx - 1) / 2
    X = np.zeros((1, nx, nx), dtype=int)
    X[0, ii, ii] = 1
    model = ElasticFESimulation(elastic_modulus=(1., 10.), poissons_ratio=(0.3, 0.3))
    strains = model.get_response(X, slice(None))
    solution = (1.518e-1, -1.672e-2, 0.)
    assert np.allclose(strains[0, ii, ii], solution, rtol=1e-3)

def test_ElasticFESimulation_3D():
    nx = 4
    ii = (nx - 1) / 2
    X = np.zeros((1, nx, nx, nx), dtype=int)
    X[0, :, ii] = 1
    model = ElasticFESimulation(elastic_modulus=(1., 10.), poissons_ratio=(0., 0.))
    strains = model.get_response(X, slice(None))
    solution = [1., 0., 0., 0., 0., 0.]
    assert np.allclose([np.mean(strains[0,...,i]) for i in range(6)], solution)

def get_delta_data(nx, ny):

    return make_elasticFEstrain_delta(elastic_modulus=(1, 1.1), 
                                      poissons_ratio=(0.3, 0.3), 
                                      size=(nx, ny),
                                      strain_index=slice(None))

def get_random_data(nx, ny):
    np.random.seed(8)
    return make_elasticFEstrain_random(elastic_modulus=(1., 1.1),
                                poissons_ratio=(0.3, 0.3), n_samples=1,
                                size=(nx, ny), strain_index=slice(None))

def rollzip(*args):
    return zip(*tuple(np.rollaxis(x, -1) for x in args))

def test_MKSelastic_delta():
    nx, ny = 21, 21
    X, y_prop = get_delta_data(nx, ny)
    basis = DiscreteIndicatorBasis(n_states=2)
        
    for y_test in np.rollaxis(y_prop, -1):
        model = MKSRegressionModel(basis)
        model.fit(X, y_test)
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y_test, rtol=1e-3, atol=1e-3)

def test_MKSelastic_random():
    nx, ny = 21, 21
    i = 3
    X_delta, strains_delta = get_delta_data(nx, ny)
    X_test, strains_test = get_random_data(nx, ny)
    basis = DiscreteIndicatorBasis(n_states=2)
        
    for y_delta, y_test in rollzip(strains_delta, strains_test):
        model = MKSRegressionModel(basis)
        model.fit(X_delta, y_delta)
        y_pred = model.predict(X_test)
        assert np.allclose(y_pred[0, i:-i], y_test[0, i:-i], rtol=1e-2, atol=6.1e-3)

def test_resize_pred():
    nx, ny = 21, 21
    i = 3
    resize = 3
    X_delta, strains_delta = get_delta_data(nx, ny)
    X_test, strains_test = get_random_data(nx, ny)
    X_big_test, strains_big_test = get_random_data(resize * nx, resize * ny)
    basis = DiscreteIndicatorBasis(n_states=2)
    
    for y_delta, y_test, y_big_test in rollzip(strains_delta, strains_test, strains_big_test):
        model = MKSRegressionModel(basis)
        model.fit(X_delta, y_delta)
        y_pred = model.predict(X_test)
        assert np.allclose(y_pred[0, i:-i], y_test[0, i:-i], rtol=1e-2, atol=6.1e-3)
        model.resize_coeff((resize * nx, resize * ny))
        y_big_pred = model.predict(X_big_test)
        assert np.allclose(y_big_pred[0, resize * i:-i * resize], y_big_test[0, resize * i:-i * resize],  rtol=1e-2, atol=6.1e-2)
        
def test_resize_coeff():
    nx, ny = 21, 21
    resize = 3
    X_delta, strains_delta = get_delta_data(nx, ny)
    X_big_delta, strains_big_delta =  get_delta_data(resize * nx, resize * ny)
    basis = DiscreteIndicatorBasis(n_states=2)
       
    for y_delta, y_big_delta in rollzip(strains_delta, strains_big_delta):
        model = MKSRegressionModel(basis)
        big_model = MKSRegressionModel(basis)
        model.fit(X_delta, y_delta)
        big_model.fit(X_big_delta, y_big_delta)
        model.resize_coeff((resize * nx, resize * ny))
        assert np.allclose(model.coeff, big_model.coeff, rtol=1e-2, atol=2.1e-3)
    
def test_multiphase():
    L = 21
    i = 3
    elastic_modulus = (80, 100, 120)
    poissons_ratio = (0.3, 0.3, 0.3)
    macro_strain = 0.02
    size = (L, L)

    X_delta, strains_delta = make_elasticFEstrain_delta(elastic_modulus=elastic_modulus,
                                                        poissons_ratio=poissons_ratio,
                                                        size=size, macro_strain=macro_strain)
    basis = DiscreteIndicatorBasis(len(elastic_modulus))
    MKSmodel = MKSRegressionModel(basis)
    MKSmodel.fit(X_delta, strains_delta)
    np.random.seed(99)
    X, strain = make_elasticFEstrain_random(n_samples=1, elastic_modulus=elastic_modulus,
                                   poissons_ratio=poissons_ratio, size=size, 
                                   macro_strain=macro_strain)
    strain_pred = MKSmodel.predict(X)
    print strain[0]
    print strain_pred[0]
    assert np.allclose(strain_pred[0, i:-i], strain[0, i:-i], rtol=1e-2, atol=6.1e-3)

    def test_cahnHilliard():
        n_samples = 100
        n_spaces = 20
        dt = 1e-3
        np.random.seed(0 )
        X, y = make_cahnHilliard(n_samples=n_samples, size=(n_spaces, n_spaces), dt=dt)
        model = MKSRegressionModel(n_states=10)
        model.fit(X, y)
        X_test = np.array([np.random.random((n_spaces, n_spaces)) for i in range(1)])
        CHSim = CahnHilliardSimulation(dt=dt)
        y_test = CHSim.get_response(X_test)
        y_pred = model.predict(X_test)
        assert mse(y_test, y_pred) < 0.03

if __name__ == '__main__':
    test_MKSelastic_delta()
