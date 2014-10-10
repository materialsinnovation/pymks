import numpy as np


def test_elastic_FE_simulation_3D():
    from pymks.datasets.elastic_FE_simulation import ElasticFESimulation
    nx = 5
    ii = (nx - 1) / 2
    X = np.zeros((1, nx, nx, nx), dtype=int)
    X[0, :, ii] = 1
    model = ElasticFESimulation(elastic_modulus=(1., 10.),
                                poissons_ratio=(0., 0.))
    model.run(X)
    solution = [1., 0., 0., 0., 0., 0.]
    assert np.allclose([np.mean(model.strain[0, ..., i]) for i in range(6)],
                       solution)


def test_elastic_FE_simulation_3D_BCs():
    from pymks.datasets.elastic_FE_simulation import ElasticFESimulation
    np.random.seed(8)
    N = 4
    X = np.random.randint(2, size=(1, N, N, N))
    macro_strain = 0.1
    sim = ElasticFESimulation((10.0, 1.0), (0.3, 0.3), macro_strain=0.1)
    sim.run(X)
    u = sim.displacement[0]
    ## Check the left/right offset
    assert np.allclose(u[-1, ..., 0] - u[0, ..., 0], N * macro_strain)
    ## Check the left/right y-periodicity
    assert np.allclose(u[0, ..., 1], u[-1, ..., 1])


def get_delta_data(nx, ny):
    from pymks.datasets import make_elastic_FE_strain_delta
    return make_elastic_FE_strain_delta(elastic_modulus=(1, 1.1),
                                        poissons_ratio=(0.3, 0.3),
                                        size=(nx, ny))


def get_random_data(nx, ny):
    from pymks.datasets import make_elastic_FE_strain_random
    np.random.seed(8)
    return make_elastic_FE_strain_random(elastic_modulus=(1., 1.1),
                                         poissons_ratio=(0.3, 0.3),
                                         n_samples=1,
                                         size=(nx, ny))


def roll_zip(*args):
    return list(zip(*tuple(np.rollaxis(x, -1) for x in args)))


def test_MKS_elastic_delta():
    from pymks import MKSRegressionModel
    from pymks.bases import DiscreteIndicatorBasis
    nx, ny = 21, 21
    X, y_test = get_delta_data(nx, ny)
    basis = DiscreteIndicatorBasis(n_states=2)
    model = MKSRegressionModel(basis)
    model.fit(X, y_test)
    y_pred = model.predict(X)
    assert np.allclose(y_pred, y_test, rtol=1e-3, atol=1e-3)


def test_MKS_elastic_random():
    from pymks import MKSRegressionModel
    from pymks.bases import DiscreteIndicatorBasis
    nx, ny = 21, 21
    X_delta, y_delta = get_delta_data(nx, ny)
    X_test, y_test = get_random_data(nx, ny)
    basis = DiscreteIndicatorBasis(n_states=2)
    model = MKSRegressionModel(basis)
    model.fit(X_delta, y_delta)
    y_pred = model.predict(X_test)
    assert np.allclose(y_pred, y_test, rtol=1e-2, atol=6.1e-3)


def test_resize_pred():
    from pymks import MKSRegressionModel
    from pymks.bases import DiscreteIndicatorBasis

    nx, ny = 21, 21
    resize = 3
    X_delta, y_delta = get_delta_data(nx, ny)
    X_test, y_test = get_random_data(nx, ny)
    X_big_test, y_big_test = get_random_data(resize * nx, resize * ny)
    basis = DiscreteIndicatorBasis(n_states=2)

    model = MKSRegressionModel(basis)
    model.fit(X_delta, y_delta)
    y_pred = model.predict(X_test)
    assert np.allclose(y_pred, y_test, rtol=1e-2, atol=6.1e-3)
    model.resize_coeff((resize * nx, resize * ny))
    y_big_pred = model.predict(X_big_test)
    assert np.allclose(y_big_pred, y_big_test, rtol=1e-2, atol=6.1e-2)


def test_resize_coeff():
    from pymks import MKSRegressionModel
    from pymks.bases import DiscreteIndicatorBasis

    nx, ny = 21, 21
    resize = 3
    X_delta, y_delta = get_delta_data(nx, ny)
    X_big_delta, y_big_delta = get_delta_data(resize * nx, resize * ny)
    basis = DiscreteIndicatorBasis(n_states=2)

    model = MKSRegressionModel(basis)
    big_model = MKSRegressionModel(basis)
    model.fit(X_delta, y_delta)
    big_model.fit(X_big_delta, y_big_delta)
    model.resize_coeff((resize * nx, resize * ny))
    assert np.allclose(model.coeff, big_model.coeff,
                       rtol=1e-2, atol=2.1e-3)


def test_multiphase_FE_strain():
    from pymks import MKSRegressionModel
    from pymks.datasets import make_elastic_FE_strain_delta
    from pymks.datasets import make_elastic_FE_strain_random
    from pymks.bases import DiscreteIndicatorBasis

    L = 21
    i = 3
    elastic_modulus = (80, 100, 120)
    poissons_ratio = (0.3, 0.3, 0.3)
    macro_strain = 0.02
    size = (L, L)

    X_delta, strains_delta = \
        make_elastic_FE_strain_delta(elastic_modulus=elastic_modulus,
                                     poissons_ratio=poissons_ratio,
                                     size=size, macro_strain=macro_strain)
    basis = DiscreteIndicatorBasis(len(elastic_modulus))
    MKSmodel = MKSRegressionModel(basis)
    MKSmodel.fit(X_delta, strains_delta)
    np.random.seed(99)
    X, strain = make_elastic_FE_strain_random(n_samples=1,
                                              elastic_modulus=elastic_modulus,
                                              poissons_ratio=poissons_ratio,
                                              size=size,
                                              macro_strain=macro_strain)
    strain_pred = MKSmodel.predict(X)

    assert np.allclose(strain_pred[0, i:-i], strain[0, i:-i],
                       rtol=1e-2, atol=6.1e-3)


def test_cahn_hilliard():
    from pymks.datasets.cahn_hilliard_simulation import CahnHilliardSimulation
    from pymks.datasets import make_cahn_hilliard
    from sklearn import metrics
    from pymks import MKSRegressionModel
    from pymks import ContinuousIndicatorBasis

    mse = metrics.mean_squared_error
    n_samples = 100
    n_spaces = 20
    dt = 1e-3
    np.random.seed(0)
    X, y = make_cahn_hilliard(n_samples=n_samples,
                              size=(n_spaces, n_spaces), dt=dt)
    basis = ContinuousIndicatorBasis(10, [-1, 1])
    model = MKSRegressionModel(basis)
    model.fit(X, y)
    X_test = np.array([np.random.random((n_spaces,
                                         n_spaces)) for i in range(1)])
    CHSim = CahnHilliardSimulation(dt=dt)
    CHSim.run(X_test)
    y_test = CHSim.response
    y_pred = model.predict(X_test)
    assert mse(y_test[0], y_pred[0]) < 0.03

if __name__ == '__main__':
    test_MKS_elastic_delta()
