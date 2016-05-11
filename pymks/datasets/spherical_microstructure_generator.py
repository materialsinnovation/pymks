from .base_microstructure_generator import _BaseMicrostructureGenerator
import numpy as np


class SphericalMicrostructureGenerator(_BaseMicrostructureGenerator):

    """
    Generates n_samples number of a periodic random spherical
    particles.  The particles have all the same phase and are in a
    matrix of a different phase (n_phase is always 2). The box is of
    size equal to size. The grain_size argument controls the size of
    the particles.

    In 1D.

    >>> size = (3, 3)
    >>> n_samples = 1
    >>> generator = SphericalMicrostructureGenerator(
    ...     n_samples=n_samples,
    ...     size=size,
    ...     n_particles=4,
    ...     grain_size=1.,
    ...     seed=10)
    >>> assert np.allclose(generator.generate(),
    ...                    [[[1, 0, 1], [1, 0, 1], [1, 0, 1]]])

    In 2D.

    >>> size = (3,)
    >>> n_samples = 1
    >>> generator = SphericalMicrostructureGenerator(
    ...     n_samples=n_samples,
    ...     size=size,
    ...     n_particles=2,
    ...     grain_size=2.,
    ...     seed=10)
    >>> assert np.allclose(generator.generate(), [[1, 1, 1]])

    In 3D.

    >>> size = (3, 3, 3)
    >>> n_samples = 1
    >>> generator = SphericalMicrostructureGenerator(
    ...     n_samples=n_samples,
    ...     size=size,
    ...     n_particles=2,
    ...     grain_size=2.,
    ...     seed=10)
    >>> X = [[[[0, 0, 0], [1, 1, 1], [0, 0, 0]],
    ...       [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ...       [[0, 0, 0], [1, 1, 1], [0, 0, 0]]]]
    >>> assert np.allclose(generator.generate(), X)

    """

    def __init__(self, n_samples=1, size=(21, 21), n_particles=2,
                 grain_size=None, seed=3):
        """
        Instantiate a SphericalMicrostructureGenerator.

        Args:
          n_samples: number of samples to be generated
          size: size of samples
          n_particles: number of particles in the microstructures
          radius: radius of the particles
          seed: seed for random number generator

        Returns:
          n_samples number of a periodic random microstructure with size equal
          to size and with n_phases number of phases.
        """
        super(SphericalMicrostructureGenerator,
              self).__init__(n_samples=n_samples, size=size, n_phases=2,
                             grain_size=grain_size, seed=seed)
        self.n_particles = n_particles
        self.average_radius = self.grain_size / 2

    def _remove_overlaps(self, position, radius, size):
        r"""
        Remove points that overlap in a periodic box.

        >>> generator = SphericalMicrostructureGenerator()
        >>> size = np.array([4, 2])
        >>> position = np.array([[0.5, 0.5], [2., 0], [3., 1.]])
        >>> radius = np.array([0.5, 0.01, 0.5])
        >>> position_, radius_ = generator._remove_overlaps(position,
        ...                                                 radius, size)
        >>> assert np.allclose(position[1:], position_)
        >>> assert np.allclose(radius[1:], radius_)

        Args:
          position: array of points of shape (N, dim)
          radius: array of radii of shape (N,)
          size: size of box, array such that len(size) = dim

        Returns:
          return tuple of points and radii that do not overlap

        """

        dist_matrix = self._periodic_distance(
            position[:, None], position[None], size)
        radius_matrix = radius[None, :] + radius[:, None]
        mask_matrix = dist_matrix < radius_matrix
        mask_matrix = np.triu(mask_matrix, k=1)
        mask = np.any(mask_matrix, axis=1)
        return position[~mask], radius[~mask]

    def _periodic_distance(self, X, Y, size):
        r"""
        Calculate the periodic distance between points in a box. X and
        Y must be broadcastable.

        >>> generator = SphericalMicrostructureGenerator()
        >>> size = np.array([4, 2])
        >>> X = np.array([[0.5, 0.5], [2., 0]])
        >>> Y = np.array([[0., 0.], [1.5, 0.5], [3., 1.]])
        >>> r = np.sqrt(2.) / 2.
        >>> d = [[r, 1], [1, r], [r, 1]]
        >>> assert np.allclose(generator._periodic_distance(X[None],
        ...                                                 Y[:, None],
        ...                                                 size), d)

        Args:
          X: array of (..., dim)
          Y: array of (..., dim)
          size: size of box, array such that len(size) = dim

        Returns:
          Periodic distances between points in array X and Y, an array
          shaped according to the standard broadcast rules for X and Y.

        """
        a = (X - Y) % (size - 1)
        b = (Y - X) % (size - 1)
        d = np.minimum(a, b)
        return np.sqrt(np.sum(d**2, axis=-1))

    def _generate_sample(self):
        size = np.array(self.size)
        dim = len(size)
        particle_positions = np.random.random(
            (self.n_particles, dim)) * (size - 1)[None]
        radius = np.random.normal(
            self.average_radius, self.average_radius / 3., self.n_particles)
        radius = radius * np.sign(radius)
        particle_positions, radius = self._remove_overlaps(
            particle_positions, radius, size)
        xyz = (np.indices(size) + 0.5).swapaxes(0, -1)
        X_ = radius > self._periodic_distance(
            particle_positions[[None] * dim], xyz[..., None, :], size)
        return np.any(X_, axis=-1).astype(int)

    def generate(self):
        """
        Generates the microstructure.

        Returns:
          periodic microstructure
        """
        return np.array([self._generate_sample()
                         for _ in range(self.n_samples)])
