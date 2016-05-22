from .cahn_hilliard_simulation import CahnHilliardSimulation
from .microstructure_generator import MicrostructureGenerator
from pymks import DiscreteIndicatorBasis, MKSRegressionModel
import numpy as np

__all__ = ['make_delta_microstructures', 'make_elastic_FE_strain_delta',
           'make_elastic_FE_strain_random', 'make_cahn_hilliard',
           'make_microstructure', 'make_checkerboard_microstructure',
           'make_elastic_stress_random']


def make_elastic_FE_strain_delta(elastic_modulus=(100, 150),
                                 poissons_ratio=(0.3, 0.3),
                                 size=(21, 21), macro_strain=0.01):
    """Generate delta microstructures and responses

    Simple interface to generate delta microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The length of `elastic_modulus` and
    `poissons_ratio` indicates the number of phases in the
    microstructure. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    Args:
        elastic_modulus (list, optional): elastic moduli for the phases
        poissons_ratio (list, optional): Poisson's ratios for the phases
        size (tuple, optional): size of the microstructure
        macro_strain (float, optional): Scalar for macroscopic strain applied
        strain_index (int, optional): interger value to return a particular
            strain field. 0 returns exx, 1 returns eyy, etc. To return all
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

    Args:
        n_phases (int, optional): number of phases
        size (tuple, optional): dimension of microstructure

    Returns:
        delta microstructures for the system of shape
        (n_samples, n_x, ...)

    Example

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

    """
    shape = (n_phases, n_phases) + size
    center = tuple((np.array(size) - 1) / 2)
    X = np.zeros(shape=shape, dtype=int)
    X[:] = np.arange(n_phases)[(slice(None), None) + (None,) * len(size)]
    X[(slice(None), slice(None)) + center] = np.arange(n_phases)
    mask = ~np.identity(n_phases, dtype=bool)
    return X[mask]


def make_elastic_FE_strain_random(n_samples=1, elastic_modulus=(100, 150),
                                  poissons_ratio=(0.3, 0.3), size=(21, 21),
                                  macro_strain=0.01):
    """Generate random microstructures and responses

    Simple interface to generate random microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    Args:
        n_samples (int, optional): number of samples
        elastic_modulus (list, optional): elastic moduli for the phases
        poissons_ratio (list, optional): Poisson's ratios for the phases
        size (tuple, optional): size of the microstructure
        macro_strain (float, optional): Scalar for macroscopic strain applied
        strain_index (int, optional): interger value to return a particular
            strain field. 0 returns exx, 1 returns eyy, etc. To return all
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

    Args:
        n_samples (int, optional): number of microstructure samples
        size (tuple, optional): size of the microstructure
        dx (float, optional): grid spacing
        dt (float, optional): timpe step size
        width (float, optional): interface width between phases.
        n_steps (int, optional): number of time steps used

    Returns:
        Array representing the microstructures at n_steps ahead of 'X'

    Example

    >>> X, y = make_cahn_hilliard(n_samples=1, size=(6, 6))

    `X` is the initial concentration fields, and `y` is the
    strain response fields (the concentration after one time step).

    """
    CHsim = CahnHilliardSimulation(dx=dx, dt=dt, gamma=width ** 2)

    X0 = 2 * np.random.random((n_samples,) + size) - 1
    X = X0.copy()
    for ii in range(n_steps):
        CHsim.run(X)
        X = CHsim.response
    return X0, X


def make_microstructure(n_samples=10, size=(101, 101), n_phases=2,
                        grain_size=None, seed=10, volume_fraction=None,
                        percent_variance=None):
    """
    Constructs microstructures for an arbitrary number of phases
    given the size of the domain, and relative grain size.

    Args:
        n_samples (int, optional): number of samples
        size (tuple, optional): dimension of microstructure
        n_phases (int, optional): number of phases
        grain_size (tuple, optional): effective dimensions of grains
        seed (int, optional): seed for random number microstructureGenerator
        volume_fraction(tuple, optional): specify the volume fraction of each
            phase
        percent_variance(int, optional): varies the volume fraction of the
            microstructure up to this percentage

    Returns:
        microstructures for the system of shape (n_samples, n_x, ...)

    Example

    >>> n_samples, n_phases = 1, 2
    >>> size, grain_size = (3, 3), (1, 1)
    >>> Xtest = np.array([[[0, 1, 0],
    ...                [0, 0, 0],
    ...                [0, 1, 1]]])
    >>> X = make_microstructure(n_samples=n_samples, size=size,
    ...                         n_phases=n_phases, grain_size=grain_size,
    ...                         seed=0)

    >>> assert(np.allclose(X, Xtest))

    """
    if grain_size is None:
        grain_size = np.array(size) / 3.
    if np.sum(np.array(grain_size) > np.array(size)):
        raise RuntimeError('grain_size must be smaller than size')
    MS = MicrostructureGenerator(n_samples=n_samples, size=size,
                                 n_phases=n_phases, grain_size=grain_size,
                                 seed=seed, volume_fraction=volume_fraction,
                                 percent_variance=percent_variance)
    return MS.generate()


def make_checkerboard_microstructure(square_size, n_squares):
    """
    Constructs a checkerboard_microstructure with the `square_size` by
    `square_size` size squares and on a `n_squares` by `n_squares`

    Args:
        square_size (int): length of the side of one square in the
            checkerboard.
        n_squares (int): number of squares along on size of the checkerboard.

    Returns:
        checkerboard microstructure with shape of (1, square_size * n_squares,
        square_size * n_squares)

    Example

    >>> square_size, n_squares = 2, 2
    >>> Xtest = np.array([[[0, 0, 1, 1],
    ...                    [0, 0, 1, 1],
    ...                    [1, 1, 0, 0],
    ...                    [1, 1, 0, 0]]])
    >>> X = make_checkerboard_microstructure(square_size, n_squares)
    >>> assert(np.allclose(X, Xtest))

    """

    L = n_squares * square_size
    X = np.ones((2 * square_size, 2 * square_size), dtype=int)
    X[:square_size, :square_size] = 0
    X[square_size:, square_size:] = 0
    return np.tile(X, (int((n_squares + 1) / 2),
                   int((n_squares + 1) / 2)))[None, :L, :L]


def make_elastic_stress_random(n_samples=[10, 10], elastic_modulus=(100, 150),
                               poissons_ratio=(0.3, 0.3), size=(21, 21),
                               macro_strain=0.01, grain_size=[(3, 3), (9, 9)],
                               seed=10, volume_fraction=None,
                               percent_variance=None):
    """
    Generates microstructures and their macroscopic stress values for an
    applied macroscopic strain.

    Args:
        n_samples (int, optional): number of samples
        elastic_modulus (tuple, optional): list of elastic moduli for the
            different phases.
        poissons_ratio (tuple, optional): list of poisson's ratio values for
            the phases.
        size (tuple, optional): size of the microstructures
        macro_strain (tuple, optional): macroscopic strain applied to the
            sample.
        grain_size (tuple, optional): effective dimensions of grains
        seed (int, optional): seed for random number generator
        volume_fraction(tuple, optional): specify the volume fraction of
            each phase
        percent_variance(int, optional): Only used if volume_fraction is
            specified. Randomly varies the volume fraction of the
            microstructure.


    Returns:
        array of microstructures with dimensions (n_samples, n_x, ...) and
        effective stress values

    """
    if not isinstance(grain_size[0], (list, tuple, np.ndarray)):
        grain_size = (grain_size,)
    if not isinstance(n_samples, (list, tuple, np.ndarray)):
        n_samples = (n_samples,)
    if volume_fraction is None:
        volume_fraction = (None,) * len(n_samples)
    vf_0 = volume_fraction[0]
    if not isinstance(vf_0, (list, tuple, np.ndarray)) and vf_0 is not None:
        volume_fraction = (volume_fraction,)
    if not isinstance(size, (list, tuple, np.ndarray)) or len(size) > 3:
        raise RuntimeError('size must have length of 2 or 3')
    [RuntimeError('dimensions of size and grain_size are not the same.')
     for grains in grain_size if len(size) != len(grains)]
    if vf_0 is not None:
        [RuntimeError('dimensions of size and grain_size are not the same.')
         for volume_frac in volume_fraction
         if len(elastic_modulus) != len(volume_frac)]
    if len(elastic_modulus) != len(poissons_ratio):
        raise RuntimeError('length of elastic_modulus and poissons_ratio are \
                           not the same.')
    X_cal, y_cal = make_elastic_FE_strain_delta(elastic_modulus,
                                                poissons_ratio, size,
                                                macro_strain)
    n_states = len(elastic_modulus)
    basis = DiscreteIndicatorBasis(n_states)
    model = MKSRegressionModel(basis=basis)
    model.fit(X_cal, y_cal)
    X = np.concatenate([make_microstructure(n_samples=sample, size=size,
                                            n_phases=n_states,
                                            grain_size=gs, seed=seed,
                                            volume_fraction=vf,
                                            percent_variance=percent_variance)
                        for vf, gs, sample in zip(volume_fraction,
                                                  grain_size, n_samples)])
    X_ = basis.discretize(X)
    index = tuple([None for i in range(len(size) + 1)]) + (slice(None),)
    modulus = np.sum(X_ * np.array(elastic_modulus)[index], axis=-1)
    y_stress = model.predict(X) * modulus
    return X, np.average(y_stress.reshape(len(y_stress), -1), axis=1)
