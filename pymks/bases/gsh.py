import numpy as np
import gsh_hex_tri_L0_16 as gsh_hex
import gsh_cub_tri_L0_16 as gsh_cub
import gsh_tri_tri_L0_13 as gsh_tri
from .abstract import _AbstractMicrostructureBasis


class GSHBasis(_AbstractMicrostructureBasis):

    r"""
    Discretize a continuous field into continuous local states using a
    Generalized Spherical Harmonic (GSH) basis such that,

    .. math::

       \frac{1}{\Delta} \int_s m(g, x) dx =
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

    >>> gsh_basis = GSHBasis(n_states=[3], domain='squishy') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    RuntimeError: invalid crystal symmetry

    >>> gsh_basis = GSHBasis(n_states=[3], domain='hex') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    RuntimeError: invalid crystal symmetry
    """

    def __init__(self, n_states=np.arange(15), domain=None):
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

        self.n_states = n_states
        if isinstance(self.n_states, int):
            self.n_states = np.arange(n_states)
        if domain in [None, 'triclinic']:
            self.domain = 'triclinic'
            self._symmetry = gsh_tri
        elif domain in ['hexagonal']:
            self.domain = 'hexagonal'
            self._symmetry = gsh_hex
        elif domain in ['cubic']:
            self.domain = 'cubic'
            self._symmetry = gsh_cub
        else:
            raise RuntimeError("invalid crystal symmetry")
        full_indx = self._symmetry.gsh_basis_info()
        self.basis_indices = full_indx[self.n_states, :]

    def check(self, X):
        """Warns the user if Euler angles apear to be defined in degrees
        instead of radians"""
        if (np.min(X) < -90.) or (np.max(X) > 90.):
            print "Warning: X may be defined in degrees instead of radians"

    def _shape_check(self, X, y):
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
        if not len(y.shape) > 1:
            raise RuntimeError("The shape of y is incorrect.")
        if y.shape != X.shape[:-1]:
            raise RuntimeError("X and y must have the same number of " +
                               "samples and microstructure shape.")
        if X.shape[-1] != 3:
            raise RuntimeError("X must have 3 continuous local states " +
                               "(euler angles)")

    def _pred_shape(self, X):
        """
        Function to describe the expected output shape of a given
        microstructure X.
        """
        return X.shape[:-1]  # X has Euler angles, while output is scalar

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ..., 3)`
                shaped array where `n_samples` is the number of samples,
                `n_x` is the spatial discretization and the last dimension
                contains the Bunge Euler angles.
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
        return self._symmetry.gsh_eval(X, self.n_states)

    def _reshape_feature(self, X, size):
        """
        Helper function used to check the shape of the microstructure,
        and change to appropriate shape.

        Args:
            X: The microstructure, an `(n_samples, n_x, ...)` shaped array
                where `n_samples` is the number of samples and `n_x` is thes
                patial discretization.

        Returns:
            microstructure with shape (n_samples, size)
        """
        new_shape = (X.shape[0],) + size + (X.shape[-1],)
        return X.reshape(new_shape)
