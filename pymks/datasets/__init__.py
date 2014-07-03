import numpy as np
from pymks.datasets.elasticFESimulation import ElasticFESimulation

__all__ = ['make_delta_microstructures', 'make_elasticFEstrain_delta',
           'make_elasticFEstrain_random']

def make_elasticFEstrain_delta(elastic_modulus, poissons_ratio, 
                               size, macro_strain=1.0, strain_index=0):
    """Generate delta microstructures and responses

    Simple interface to generate delta microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The length of `elastic_modulus` and
    `poissons_ratio` indicates the number of phases in the
    microstructure. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    >>> elastic_modulus = (1., 2.)
    >>> poissons_ratio = (0.3, 0.3)
    >>> X, y = make_elasticFEstrain_delta(elastic_modulus=elastic_modulus,
    ...                                   poissons_ratio=poissons_ratio,
    ...                                   size=(5, 5)) #doctest: +ELLIPSIS
    sfepy: ...

    `X` is the delta microstructures, and `y` is the
    strain response fields.

    Args:
      elastic_modulus: list of elastic moduli for the phases
      poissons_ratio: list of Poisson's ratios for the phases
      size: size of the microstructure
      macro_strain: Scalar for macroscopic strain applied 
      strain_index: interger value to return a particular strain
        field.  0 returns exx, 1 returns eyy, etc. To return all
        strain fields set strain_index equal to slice(None).

    Returns:
      tuple containing delta microstructures and their strain fields

    """
    FEsim = ElasticFESimulation(elastic_modulus=elastic_modulus,
                                poissons_ratio=poissons_ratio,
                                macro_strain=macro_strain)

    X = make_delta_microstructures(len(elastic_modulus), size=size)
    return X, FEsim.get_response(X, strain_index=strain_index)

def make_delta_microstructures(Nphases, size):
    """Constructs delta microstructures

    Constructs delta microstructures for an arbitrary number of phases
    given the size of the domain.

    >>> X = np.array([[[[0, 0, 0],
    ...                 [0, 0, 0],
    ...                 [0, 0, 0]],
    ...                [[0, 0, 0],
    ...                 [0, 1, 0],
    ...                 [0, 0, 0]],
    ...                [[0, 0, 0],
    ...                 [0, 0, 0],
    ...                 [0, 0, 0]]],
    ...               [[[1, 1, 1],
    ...                 [1, 1, 1],
    ...                 [1, 1, 1]],
    ...                [[1, 1, 1],
    ...                 [1, 0, 1],
    ...                 [1, 1, 1]],
    ...                [[1, 1, 1],
    ...                 [1, 1, 1],
    ...                 [1, 1, 1]]]])

    >>> assert(np.allclose(X, make_delta_microstructures(2, size=(3, 3, 3))))

    Args:
        Nphases: number of phases
        size: dimension of microstructure

    Returns:
        delta microstructures for the system of shape
        (Nsamples, Nx, Ny, ...)

    """
    shape = (Nphases, Nphases) + size
    center = tuple((np.array(size) - 1) / 2)
    X = np.zeros(shape=shape, dtype=int)
    X[:] = np.arange(Nphases)[(slice(None), None) + (None,) * len(size)]
    X[(slice(None), slice(None)) + center] = np.arange(Nphases)
    mask = ~np.identity(Nphases, dtype=bool)
    return X[mask]

def make_elasticFEstrain_random(n_samples, elastic_modulus, poissons_ratio,
                                size, macro_strain=1.0, strain_index=0):
    """Generate delta microstructures and responses

    Simple interface to generate delta microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The length of `elastic_modulus` and
    `poissons_ratio` indicates the number of phases in the
    microstructure. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    >>> elastic_modulus = (1., 2.)
    >>> poissons_ratio = (0.3, 0.3)
    >>> X, y = make_elasticFEstrain_random(n_samples=1,
    ...                                    elastic_modulus=elastic_modulus,
    ...                                    poissons_ratio=poissons_ratio,
    ...                                    size=(5, 5)) #doctest: +ELLIPSIS
    sfepy: ...

    `X` is the delta microstructures, and `y` is the
    strain response fields.

    Args:
      elastic_modulus: list of elastic moduli for the phases
      poissons_ratio: list of Poisson's ratios for the phases
      n_samples: number of microstructure samples
      size: size of the microstructure
      macro_strain: Scalar for macroscopic strain applied
      strain_index: interger value to return a particular strain
        field.  0 returns exx, 1 returns eyy, etc. To return all
        strain fields set strain_index equal to slice(None).

    Returns:
      tuple containing delta microstructures and their strain fields

    """
    FEsim = ElasticFESimulation(elastic_modulus=elastic_modulus,
                                poissons_ratio=poissons_ratio,
                                macro_strain=macro_strain)

    X = np.random.randint(len(elastic_modulus), size=((n_samples,)+size))
    return X, FEsim.get_response(X, strain_index=strain_index)
