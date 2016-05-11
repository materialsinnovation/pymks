from .gsh_functions import hex_eval
from .gsh_functions import cub_eval
from .gsh_functions import tri_eval
from .gsh_functions import hex_basis_info
from .gsh_functions import cub_basis_info
from .gsh_functions import tri_basis_info
from .imag_ffts import _ImagFFTBasis
import numpy as np


class GSHBasis(_ImagFFTBasis):

    r"""
    Discretize a continuous field made up three Euler angles (in radians) used
    to represent crystal orientation into continuous local states using the
    Generalized Spherical Harmonic (GSH) basis. This basis uses the following
    equation to discretize the orientation field.

    .. math::

       \frac{1}{\Delta x} \int_s m(g, x) dx =
       \sum_{l, m, n} m[l, \tilde{m}, n, s] T_l^{\tilde{m}n}(g)

    where the :math:`T_l^{\tilde{m}n}` are GSH basis functions and the
    local state space :math:`H` is mapped into the orthogonal, periodic
    domain of the GSH functions

    The mapping of :math:`H` into some desired periodic domain is done
    automatically in PyMKS by using the `domain` key work argument to
    select the desired crystal symmetry.

    >>> X = np.array([[0.1, 0.2, 0.3],
    ...               [6.5, 2.3, 3.4]])
    >>> gsh_basis = GSHBasis(n_states = [3], domain='hexagonal')
    >>> def test_gsh(x):
    ...     phi = x[:, 1]
    ...     t915 = np.cos(phi)
    ...     return 0.15e2 / 0.2e1 * t915 ** 2 - 0.5e1 / 0.2e1

    >>> assert(np.allclose(np.squeeze(gsh_basis.discretize(X)), test_gsh(X)))

    If you select an invalid crystal symmetry PyMKS will give an error

    >>> gsh_basis = GSHBasis(n_states=[3], domain='squishy')
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    RuntimeError: invalid crystal symmetry

    >>> gsh_basis = GSHBasis(n_states=[3], domain='hex') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    RuntimeError: invalid crystal symmetry
    """

    def __init__(self, n_states=15, domain=None):
        """
        Instantiate a `Basis`

        Args:
            n_states (int, array): An array of local states to be used. states
                requested. If an integer is provided, all local states up
                to that number will be used.
            domain (list, optional): indicate the desired crystal symmetry for
                the GSH. Valid choices for symmetry are "hexagonal", "cubic" or
                "triclinic" if no symmetry is desired (not specifying any
                symmetry has the same effect)
        """
        if isinstance(n_states, int):
            n_states = np.arange(n_states)
        if domain is None or domain == 'triclinic':
            self.domain = 'triclinic'
        elif domain == 'hexagonal':
            self.domain = 'hexagonal'
        elif domain == 'cubic':
            self.domain = 'cubic'
        else:
            raise RuntimeError("invalid crystal symmetry")
        full_indx = self._gsh_basis_info()
        super(GSHBasis, self).__init__(n_states=n_states, domain=self.domain)
        self.basis_indices = full_indx[self.n_states, :]

    def check(self, X):
        """Warns the user if Euler angles apear to be defined in degrees
        instead of radians"""
        if (np.min(X) < -90.) or (np.max(X) > 90.):
            Warning("X may be defined in degrees instead of radians")
        if X.shape[-1] != 3:
            raise RuntimeError('X must have 3 angles (in radians) in the ' +
                               'last dimention')

    def _check_shape(self, X_shape, y_shape):
        """
        Checks the shape of the microstructure and response data to
        ensure that they are correct.

        Firstly, the response data "y" must have a dimension to index the
        microstructure instantiation and at least one dimension to index the
        local microstructural information.

        Second, the shape of X and y must me equal except for the last
        dimension of X.

        Finally, the length of the final dimension of X must be 3.
        This is because we assume that Bunge Euler angles are assigned for
        each location in the microstructure
        """
        if not len(y_shape) > 1:
            raise RuntimeError("The shape of y is incorrect.")
        if X_shape[-1] != 3:
            raise RuntimeError("X must have 3 continuous local states " +
                               "(euler angles in radians) in the last axis.")
        if y_shape != X_shape[:-1]:
            raise RuntimeError("The X and y must have the same number of " +
                               "samples and microstructure shape.")

    def _pred_shape(self, X):
        """
        Function to describe the expected output shape of a given
        microstructure X.
        """
        _shape = X.shape[:-1]  # X has Euler angles, while output is scalar
        if len(_shape) < 2:
            _shape = (X.shape[0],) + (X.shape[1] / 3,)
        return _shape

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ..., 3)`
                shaped array where `n_samples` is the number of samples,
                `n_x` is the spatial discretization and the last dimension
                contains the Bunge Euler angles in radians.
        Returns:
            Float valued field of of Generalized Spherical Harmonics
            coefficients.

        >>> X = np.array([[0.1, 0.2, 0.3],
        ...               [6.5, 2.3, 3.4]])
        >>> gsh_basis = GSHBasis(n_states = [1])
        >>> def q(x):
        ...     phi1 = x[:, 0]
        ...     phi = x[:, 1]
        ...     phi2 = x[:, 2]
        ...     x_GSH = ((0.3e1 / 0.2e1) * (0.1e1 + np.cos(phi)) *
        ...              np.exp((-1*1j) * (phi1 + phi2)))
        ...     return x_GSH
        >>> assert(np.allclose(np.squeeze(gsh_basis.discretize(X)), q(X)))
        """
        self.check(X)
        self._select_axes(X[..., 0])
        return self._gsh_eval(X)

    def _reshape_feature(self, X, size):
        """
        Helper function used to check the shape of the microstructure,
        and change to appropriate shape.

        Args:
            X: The microstructure, an `(n_samples, n_x, ...)` shaped array
                where `n_samples` is the number of samples and `n_x` is thes
                patial discretization.
            size: the new size of the array

        Returns:
            microstructure with shape (n_samples, size)
        """
        _shape = (X.shape[0],) + size + (3,)
        return X.reshape(_shape)

    def _gsh_basis_info(self):
        """
        Returns a an array with the indices used in the GSH functions for a
        a specific crystal symmetry.
        """
        if self.domain == 'triclinic':
            return tri_basis_info()
        elif self.domain == 'hexagonal':
            return hex_basis_info()
        elif self.domain == 'cubic':
            return cub_basis_info()

    def _gsh_eval(self, X):
        """
        Discretize `X` based on crystal symmetry.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ..., 3)`
                shaped array where `n_samples` is the number of samples,
                `n_x` is the spatial discretization and the last dimension
                contains the Bunge Euler angles in radians.
        Returns:
            Float valued field of of Generalized Spherical Harmonics
            coefficients.
        """
        if self.domain == 'triclinic':
            return tri_eval(X, self.n_states)
        elif self.domain == 'hexagonal':
            return hex_eval(X, self.n_states)
        elif self.domain == 'cubic':
            return cub_eval(X, self.n_states)
