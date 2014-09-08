import numpy as np

from .cahn_hilliard_simulation import CahnHilliardSimulation
from .microstructure_generator import MicrostructureGenerator

__all__ = ['make_delta_microstructures', 'make_elastic_FE_strain_delta',
           'make_elastic_FE_strain_random',
           'make_cahn_hilliard', 'make_microstructure']


def make_elastic_FE_strain_delta(elastic_modulus=(1, 1), poissons_ratio=(1, 1),
                                 size=(21, 21), macro_strain=1.0):
    """Generate delta microstructures and responses

    Simple interface to generate delta microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The length of `elastic_modulus` and
    `poissons_ratio` indicates the number of phases in the
    microstructure. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    >>> elastic_modulus = (1., 2.)
    >>> poissons_ratio = (0.3, 0.3)
    >>> X, y = make_elastic_FE_strain_delta(elastic_modulus=elastic_modulus,
    ...                                     poissons_ratio=poissons_ratio,
    ...                                     size=(5, 5))

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
    from .elastic_FE_simulation import ElasticFESimulation
    
    FEsim = ElasticFESimulation(elastic_modulus=elastic_modulus,
                                poissons_ratio=poissons_ratio,
                                macro_strain=macro_strain)

    X = make_delta_microstructures(len(elastic_modulus), size=size)
    FEsim.run(X)
    return X, FEsim.response


def make_delta_microstructures(n_phases=2, size=(21, 21)):
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
        n_phases: number of phases
        size: dimension of microstructure

    Returns:
        delta microstructures for the system of shape
        (Nsamples, Nx, Ny, ...)

    """
    shape = (n_phases, n_phases) + size
    center = tuple((np.array(size) - 1) / 2)
    X = np.zeros(shape=shape, dtype=int)
    X[:] = np.arange(n_phases)[(slice(None), None) + (None,) * len(size)]
    X[(slice(None), slice(None)) + center] = np.arange(n_phases)
    mask = ~np.identity(n_phases, dtype=bool)
    return X[mask]


def make_elastic_FE_strain_random(n_samples=1, elastic_modulus=(1, 1), poissons_ratio=(1, 1),
                                  size=(21, 21), macro_strain=1.0):
    """Generate random microstructures and responses

    Simple interface to generate random microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    >>> elastic_modulus = (1., 2.)
    >>> poissons_ratio = (0.3, 0.3)
    >>> X, y = make_elastic_FE_strain_random(n_samples=1,
    ...                                      elastic_modulus=elastic_modulus,
    ...                                      poissons_ratio=poissons_ratio,
    ...                                      size=(5, 5))

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
    from .elastic_FE_simulation import ElasticFESimulation
    
    FEsim = ElasticFESimulation(elastic_modulus=elastic_modulus,
                                poissons_ratio=poissons_ratio,
                                macro_strain=macro_strain)

    X = np.random.randint(len(elastic_modulus), size=((n_samples, ) + size))
    FEsim.run(X)
    return X, FEsim.response


def make_cahn_hilliard(n_samples=1, size=(21, 21), dx=0.25, width=1.,
                       dt=0.001, n_steps=1):
    """Generate microstructures and responses for Cahn-Hilliard.
    Simple interface to generate random concentration fields and their
    evolution after one time step that can be used for the fit method in the
    `MKSRegressionModel`.  The following example is or a two phase
    microstructure with dimensions of `(6, 6)`.

    >>> X, y = make_cahn_hilliard(n_samples=1, size=(6, 6))

    `X` is the initial concentration fields, and `y` is the
    strain response fields (the concentration after one time step).

    Args:
      n_samples: number of microstructure samples
      size: size of the microstructure
      dx: grid spacing
      dt: timpe step size
      width: interface width between phases.
      n_steps: number of time steps used

    Returns:
      Array representing the microstructures at n_steps ahead of 'X'

    """
    CHsim = CahnHilliardSimulation(dx=dx, dt=dt, gamma=width**2)

    X0 = 2 * np.random.random((n_samples,) + size) - 1
    X = X0.copy()
    for ii in range(n_steps):
        CHsim.run(X)
        X = CHsim.response
    return X0, X


def make_microstructure(n_samples=10, size=(101, 101), n_phases=2,
                        grain_size=(33, 14), seed=10):
    """
    Constructs microstructures for an arbitrary number of phases
    given the size of the domain, and relative grain size.

    >>> n_samples, n_phases = 1, 2
    >>> size, grain_size = (3, 3), (1, 1)
    >>> Xtest = np.array([[[0, 1, 0],
    ...                [0, 0, 0],
    ...                [0, 1, 1]]])
    >>> X = make_microstructure(n_samples=n_samples, size=size,
    ...                         n_phases=n_phases, grain_size=grain_size,
    ...                         seed=0)

    >>> assert(np.allclose(X, Xtest))

    Args:
        n_samples: number of samples
        size: dimension of microstructure
        n_phases: number of phases
        grain_size: effective dimensions of grains
        seed: seed for random number microstructureGenerator

    Returns:
        microstructures for the system of shape (Nsamples, Nx, Ny, ...)

    """
    MS = MicrostructureGenerator(n_samples=n_samples, size=size,
                                  n_phases=n_phases, grain_size=grain_size,
                                  seed=seed)
    return MS.generate()
