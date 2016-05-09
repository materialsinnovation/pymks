from nose.tools import raises
from pymks import skip_sfepy

@skip_sfepy
@raises(RuntimeError)
def test_issue106():
    from pymks.datasets.elastic_FE_simulation import ElasticFESimulation
    import numpy as np
    L = 5
    elastic_modulus = (1, 2, 3)
    poissons_ratio = (0.3, 0.3, 0.3)
    size = (1, L, L)
    sim = ElasticFESimulation(elastic_modulus=elastic_modulus,
                              poissons_ratio=poissons_ratio)
    X = np.zeros(size, dtype=int)
    sim.run(X)
    X = np.ones(size, dtype=int)
    sim.run(X)
    X[0, 0, 0] = -1
    sim.run(X)

@skip_sfepy
def test_main():
    from pymks.datasets.elastic_FE_simulation import ElasticFESimulation
    import numpy as np
    X = np.zeros((1, 3, 3), dtype=int)
    X[0, :, 1] = 1

    sim = ElasticFESimulation(elastic_modulus=(1.0, 10.0),
                              poissons_ratio=(0., 0.))
    sim.run(X)
    y = sim.strain

    exx = y[..., 0]
    eyy = y[..., 1]
    exy = y[..., 2]

    assert np.allclose(exx, 1)
    assert np.allclose(eyy, 0)
    assert np.allclose(exy, 0)

    X = np.array([[[1, 0, 0, 1],
                   [0, 1, 1, 1],
                   [0, 0, 1, 1],
                   [1, 0, 0, 1]]])
    n_samples, N, N = X.shape
    macro_strain = 0.1
    sim = ElasticFESimulation((10.0,1.0), (0.3,0.3), macro_strain=0.1)
    sim.run(X)
    u = sim.displacement[0]

    assert np.allclose(u[-1,:,0] - u[0,:,0], N * macro_strain)
    assert np.allclose(u[0,:,1], u[-1,:,1])
    assert np.allclose(u[:,0], u[:,-1])

@skip_sfepy
def test_convert_properties():
    from pymks.datasets.elastic_FE_simulation import ElasticFESimulation
    import numpy as np
    model = ElasticFESimulation(elastic_modulus=(1., 2.),
                                poissons_ratio=(1., 1.))
    result = model._convert_properties(2)
    answer = np.array([[-0.5, 1. / 6.], [-1., 1. / 3.]])
    assert(np.allclose(result, answer))

@skip_sfepy
def test_get_property_array():
    from pymks.datasets.elastic_FE_simulation import ElasticFESimulation
    import numpy as np
    X2D = np.array([[[0, 1, 2, 1],
                     [2, 1, 0, 0],
                     [1, 0, 2, 2]]])
    model2D = ElasticFESimulation(elastic_modulus=(1., 2., 3.),
                                  poissons_ratio=(1., 1., 1.))
    lame = lame0, lame1, lame2 = -0.5, -1., -1.5
    mu = mu0, mu1, mu2 = 1. / 6, 1. / 3, 1. / 2
    lm = zip(lame, mu)
    X2D_property = np.array([[lm[0], lm[1], lm[2], lm[1]],
                             [lm[2], lm[1], lm[0], lm[0]],
                             [lm[1], lm[0], lm[2], lm[2]]])

    assert(np.allclose(model2D._get_property_array(X2D), X2D_property))

    model3D = ElasticFESimulation(elastic_modulus=(1., 2.),
                                  poissons_ratio=(1., 1.))
    X3D = np.array([[[0, 1],
                     [0, 0]],
                    [[1, 1],
                     [0, 1]]])
    X3D_property = np.array([[[lm[0], lm[1]],
                              [lm[0], lm[0]]],
                             [[lm[1], lm[1]],
                              [lm[0], lm[1]]]])
    assert(np.allclose(model3D._get_property_array(X3D), X3D_property))
