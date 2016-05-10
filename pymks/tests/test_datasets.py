from pymks.datasets import make_elastic_FE_strain_random
from pymks.datasets import make_elastic_FE_strain_delta
from pymks.datasets import make_elastic_stress_random
import numpy as np


def test_make_elastic_FE_strain_delta():
    elastic_modulus = (1., 2.)
    poissons_ratio = (0.3, 0.3)
    X, y = make_elastic_FE_strain_delta(elastic_modulus=elastic_modulus,
                                        poissons_ratio=poissons_ratio,
                                        size=(5, 5))


def test_make_elastic_FE_strain_random():
    elastic_modulus = (1., 2.)
    poissons_ratio = (0.3, 0.3)
    X, y = make_elastic_FE_strain_random(n_samples=1,
                                         elastic_modulus=elastic_modulus,
                                         poissons_ratio=poissons_ratio,
                                         size=(5, 5))


def test_make_elastic_stress_randome():
    X, y = make_elastic_stress_random(n_samples=1, elastic_modulus=(1, 1),
                                      poissons_ratio=(1, 1),
                                      grain_size=(3, 3), macro_strain=1.0)
    assert np.allclose(y, np.ones(y.shape))
    X, y = make_elastic_stress_random(n_samples=1, grain_size=(1, 1),
                                      elastic_modulus=(100, 200),
                                      size=(2, 2), poissons_ratio=(1, 3),
                                      macro_strain=1., seed=3)
    X_result = np.array([[[1, 1],
                          [0, 1]]])
    assert np.allclose(X, X_result)
    assert float(np.round(y, decimals=5)[0]) == 228.74696
    X, y = make_elastic_stress_random(n_samples=1, grain_size=(1, 1, 1),
                                      elastic_modulus=(100, 200),
                                      poissons_ratio=(1, 3),  seed=3,
                                      macro_strain=1., size=(2, 2, 2))
    X_result = np.array([[[1, 1],
                          [0, 0]],
                         [[1, 1],
                          [0, 0]]])
    assert np.allclose(X, X_result)
    assert np.round(y[0]).astype(int) == 150
