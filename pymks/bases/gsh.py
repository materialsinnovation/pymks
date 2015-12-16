import numpy as np
import gsh_hex_tri_L0_4_alt as GSHHex
from .abstract import _AbstractMicrostructureBasis


class GSHBasis(_AbstractMicrostructureBasis):

    r"""

    NOTE: THIS MUST BE MODIFIED FOR GSH

    Discretize a continuous field into `deg` local states using a
    Legendre polynomial basis such that,

    .. math::

       \frac{1}{\Delta} \int_s m(h, x) dx =
       \sum_0^{L-1} m[l, s] P_l(h)

    where the :math:`P_l` are Legendre polynomials and the local state space
    :math:`H` is mapped into the orthogonal domain of the Legendre polynomials

    .. math::

       -1 \le  H \le 1

    The mapping of :math:`H` into the domain is done automatically in PyMKS by
    using the `domain` key work argument.

    >>> n_states = 3
    >>> X = np.array([[0.25, 0.1],
    ...               [0.5, 0.25]])
    >>> def P(x):
    ...    x = 4 * x - 1
    ...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
    ...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
    ...    return np.rollaxis(tmp, 0, 3)
    >>> basis = LegendreBasis(n_states, [0., 0.5])
    >>> assert(np.allclose(basis.discretize(X), P(X)))

    If the microstructure local state values fall outside of the specified
    domain they will no longer be mapped into the orthogonal domain of the
    legendre polynomais.

    >>> n_states = 2
    >>> X = np.array([-1, 1])
    >>> leg_basis = LegendreBasis(n_states, domain=[0, 1])
    >>> leg_basis.discretize(X)
    Traceback (most recent call last):
    ...
    RuntimeError: X must be within the specified domain

    """

    def __init__(self, n_states=np.arange(15), domain=None):
        """
        Instantiate a `Basis`

        Args:
            n_states (int): The number of local states
            domain (list, optional): indicate the range of expected values for
                the microstructure, default is [0, n_states - 1].
        """

        if type(n_states) == int:
            self.n_states = np.arange(n_states)
            print "Warning: for an integer n_states, the GSH basis functions with" +\
                  " linear indices up to n_states will be used. To use a single " +\
                  "basis function or a set of indices assign a list, tuple or " +\
                  "array to n_states"
        else:
            self.n_states = n_states

        self.domain = domain

    def check(self, X):
        if (np.min(X) < -90.) or (np.max(X) > 90.):
            raise UserError("X may be defined in degrees instead of radians")

    def _shape_check(self, X, y):
        if not len(y.shape) > 1:
            raise RuntimeError("The shape of y is incorrect.")
        if y.shape != X.shape[:-1]:
            raise RuntimeError("X and y must have the same number of samples and microstructure shape.")
        if X.shape[-1] != 3:
            raise RuntimeError("X must have 3 continuous local states (euler angles)") 

    def _output_shape(self,X):
        """
        Function to describe the expected output shape of a given
        microstructure X.
        """
        return X.shape[:-1] #X has Euler angles, while output is scalar

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
        Returns:
            Float valued field of of Legendre polynomial coefficients.

        >>> X = np.array([[-1, 1],
        ...               [0, -1]])
        >>> leg_basis = LegendreBasis(3, [-1, 1])
        >>> def p(x):
        ...    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
        ...    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
        ...    return np.rollaxis(tmp, 0, 3)
        >>> assert(np.allclose(leg_basis.discretize(X), p(X)))

        """
        self.check(X)

        if self.domain == None:
            X_GSH = GSHHex.gsh_eval(X, self.n_states)
            # replace with vanila GSH basis functions
        elif self.domain in ['hex', 'Hex', 'hexagonal',
                             'Hexagonal', 'hcp', 'HCP']:
            X_GSH = GSHHex.gsh_eval(X, self.n_states)
        else:
            raise UserError("please select a valid crystal symmetry")

        return X_GSH

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
