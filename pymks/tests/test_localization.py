import numpy as np
from test import get_delta_data, get_random_data


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
    assert np.allclose(model.coef_, big_model.coef_,
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


def test_coeff_stablity_with_irfftn():
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
    for i in range(4):
        model.coef_
    y_big_pred = model.predict(X_big_test)
    assert np.allclose(y_big_pred, y_big_test, rtol=1e-2, atol=6.1e-2)


def test_setting_kernel():
    from pymks.datasets import make_elastic_FE_strain_delta
    from pymks import MKSLocalizationModel
    from pymks import PrimitiveBasis
    elastic_modulus = (100, 130)
    poissons_ratio = (0.3, 0.3)
    X_delta, y = make_elastic_FE_strain_delta(size=(21, 21),
                                              elastic_modulus=elastic_modulus,
                                              poissons_ratio=poissons_ratio)
    p_basis = PrimitiveBasis(2)
    model = MKSLocalizationModel(basis=p_basis)
    model.fit(X_delta, y)
    coefs = model.coef_
    model.resize_coeff((30, 30))
    model.coef_ = coefs
    assert np.allclose(model.predict(X_delta), y, atol=1e-4)

if __name__ == '__main__':
    test_resize_coeff()
