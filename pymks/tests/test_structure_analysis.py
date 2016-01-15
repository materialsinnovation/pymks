import numpy as np


def test_n_componets_from_reducer():
    from pymks import MKSStructureAnalysis
    from pymks import DiscreteIndicatorBasis
    from sklearn.manifold import LocallyLinearEmbedding
    reducer = LocallyLinearEmbedding(n_components=7)
    print reducer.n_components
    dbasis = DiscreteIndicatorBasis(n_states=3, domain=[0, 2])
    model = MKSStructureAnalysis(dimension_reducer=reducer, basis=dbasis)
    print model.n_components
    assert model.n_components == 7


def test_n_components_with_reducer():
    from pymks import MKSStructureAnalysis
    from pymks import DiscreteIndicatorBasis
    from sklearn.manifold import Isomap
    reducer = Isomap(n_components=7)
    dbasis = DiscreteIndicatorBasis(n_states=3, domain=[0, 2])
    model = MKSStructureAnalysis(dimension_reducer=reducer, basis=dbasis,
                                 n_components=9)
    assert model.n_components == 9


def test_n_components_change():
    from pymks import MKSStructureAnalysis
    from pymks import DiscreteIndicatorBasis
    dbasis = DiscreteIndicatorBasis(n_states=2)
    model = MKSStructureAnalysis(basis=dbasis)
    model.n_components = 27
    assert model.n_components == 27


def test_default_n_components():
    from pymks import MKSStructureAnalysis
    from pymks import DiscreteIndicatorBasis
    dbasis = DiscreteIndicatorBasis(n_states=2)
    model = MKSStructureAnalysis(basis=dbasis)
    assert model.n_components == 5


def test_default_dimension_reducer():
    from sklearn.decomposition import RandomizedPCA
    from pymks import MKSStructureAnalysis
    from pymks import PrimitiveBasis
    model = MKSStructureAnalysis(basis=PrimitiveBasis())
    assert isinstance(model.dimension_reducer, RandomizedPCA)


def test_default_correlations():
    from pymks import PrimitiveBasis
    from pymks import MKSStructureAnalysis
    prim_basis = PrimitiveBasis(6)
    model_prim = MKSStructureAnalysis(basis=prim_basis)
    assert model_prim.correlations == [(0, 0), (0, 1), (0, 2),
                                       (0, 3), (0, 4), (0, 5)]


def test_set_correlations():
    from pymks import PrimitiveBasis
    from pymks import MKSStructureAnalysis
    test_correlations = [(0, 0), (0, 2), (0, 4)]
    prim_basis = PrimitiveBasis(6)
    model_prim = MKSStructureAnalysis(basis=prim_basis,
                                      correlations=test_correlations)
    assert model_prim.correlations == test_correlations


def test_reshape_X():
    from pymks import MKSStructureAnalysis
    from pymks import PrimitiveBasis
    anaylzer = MKSStructureAnalysis(basis=PrimitiveBasis())
    X = np.arange(18).reshape(2, 3, 3)
    X_test = np.concatenate((np.arange(-4, 5)[None], np.arange(-4, 5)[None]))
    assert np.allclose(anaylzer._reduce_shape(X), X_test)

if __name__ == '__main__':
    test_set_correlations()
