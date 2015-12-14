import numpy as np


def test_n_componets_from_reducer():
    from pymks import MKSHomogenizationModel, DiscreteIndicatorBasis
    from sklearn.manifold import LocallyLinearEmbedding
    reducer = LocallyLinearEmbedding(n_components=7)
    dbasis = DiscreteIndicatorBasis(n_states=3, domain=[0, 2])
    model = MKSHomogenizationModel(dimension_reducer=reducer, basis=dbasis)
    assert model.n_components == 7


def test_n_components_with_reducer():
    from pymks import MKSHomogenizationModel, DiscreteIndicatorBasis
    from sklearn.manifold import SpectralEmbedding
    reducer = SpectralEmbedding(n_components=7)
    dbasis = DiscreteIndicatorBasis(n_states=3, domain=[0, 2])
    model = MKSHomogenizationModel(dimension_reducer=reducer, basis=dbasis,
                                   n_components=9)
    assert model.n_components == 9


def test_stress():
    from pymks.datasets import make_elastic_stress_random
    from pymks import MKSHomogenizationModel, DiscreteIndicatorBasis
    sample_size = 200
    grain_size = [(7, 7), (8, 3), (3, 9), (2, 2)]
    n_samples = [sample_size] * len(grain_size)
    elastic_modulus = (410, 200)
    poissons_ratio = (0.28, 0.3)
    macro_strain = 0.001
    size = (21, 21)
    X, y = make_elastic_stress_random(n_samples=n_samples, size=size,
                                      grain_size=grain_size,
                                      elastic_modulus=elastic_modulus,
                                      poissons_ratio=poissons_ratio,
                                      macro_strain=macro_strain, seed=0)
    dbasis = DiscreteIndicatorBasis(n_states=2, domain=[0, 1])
    model = MKSHomogenizationModel(basis=dbasis, n_components=3, degree=3)
    model.fit(X, y)
    test_sample_size = 1
    n_samples = [test_sample_size] * len(grain_size)
    X_new, y_new = make_elastic_stress_random(n_samples=n_samples,
                                              size=size, grain_size=grain_size,
                                              elastic_modulus=elastic_modulus,
                                              poissons_ratio=poissons_ratio,
                                              macro_strain=macro_strain,
                                              seed=8)
    y_result = model.predict(X_new)
    print np.round(y_result, decimals=2)
    print np.round(y_new, decimals=2)
    assert np.allclose(np.round(y_new, decimals=2),
                       np.round(y_result, decimals=2))


def test_n_components_change():
    from pymks import MKSHomogenizationModel, DiscreteIndicatorBasis
    dbasis = DiscreteIndicatorBasis(n_states=2)
    model = MKSHomogenizationModel(basis=dbasis)
    model.n_components = 27
    assert model.n_components == 27


def test_degree_change():
    from pymks import MKSHomogenizationModel, DiscreteIndicatorBasis
    dbasis = DiscreteIndicatorBasis(n_states=2)
    model = MKSHomogenizationModel(basis=dbasis)
    model.degree = 4
    assert model.degree == 4


def test_default_degree():
    from pymks import MKSHomogenizationModel, DiscreteIndicatorBasis
    dbasis = DiscreteIndicatorBasis(n_states=2)
    model = MKSHomogenizationModel(basis=dbasis)
    assert model.degree == 1


def test_default_n_components():
    from pymks import MKSHomogenizationModel, DiscreteIndicatorBasis
    dbasis = DiscreteIndicatorBasis(n_states=2)
    model = MKSHomogenizationModel(basis=dbasis)
    assert model.n_components == 2


def test_default_property_linker():
    from sklearn.linear_model import LinearRegression
    from pymks import MKSHomogenizationModel, PrimitiveBasis
    prim_basis = PrimitiveBasis(n_states=2)
    model = MKSHomogenizationModel(basis=prim_basis)
    assert isinstance(model.property_linker, LinearRegression)


def test_default_dimension_reducer():
    from sklearn.decomposition import RandomizedPCA
    from pymks import MKSHomogenizationModel
    model = MKSHomogenizationModel()
    assert isinstance(model.dimension_reducer, RandomizedPCA)


def test_default_correlations():
    from pymks import PrimitiveBasis
    from pymks import MKSHomogenizationModel
    prim_basis = PrimitiveBasis(6)
    model_prim = MKSHomogenizationModel(basis=prim_basis)
    assert model_prim.correlations == [(0, 0), (0, 1), (0, 2),
                                       (0, 3), (0, 4), (0, 5)]


def test_set_correlations():
    from pymks import PrimitiveBasis
    from pymks import MKSHomogenizationModel
    test_correlations = [(0, 0), (0, 2), (0, 4)]
    prim_basis = PrimitiveBasis(6)
    model_prim = MKSHomogenizationModel(basis=prim_basis,
                                        correlations=test_correlations)
    assert model_prim.correlations == test_correlations


if __name__ == '__main__':
    test_default_dimension_reducer()
