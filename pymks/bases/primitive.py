from .real_ffts import _RealFFTBasis
import numpy as np


class PrimitiveBasis(_RealFFTBasis):

    r"""
    Discretize the microstructure function into `n_states` local states such
    that:

    .. math::

       \frac{1}{\Delta x} \int_{H} \int_{s} \Lambda(h - l)
       m(h, x) dx dh = m[l, s]

    where :math:`\Lambda` is the primitive basis (also called hat basis)
    function

    .. math::

       \Lambda (h - l) = max \Bigg (1-\Bigg |\frac{h(L - 1)}{H} -
       \frac{Hl}{L-1} \Bigg|, 0\Bigg)

    A microstructure function discretized with this basis is subject to the
    following constraint

    .. math::

       \sum_{l=0}^L m[l, s] = 1

    which is equivalent of saying that every location is filled with some
    configuration of local states.

    Here is an example with 3 discrete local states in a microstructure.

    >>> X = np.array([[[1, 1, 0],
    ...                [1, 0 ,2],
    ...                [0, 1, 0]]])
    >>> assert(X.shape == (1, 3, 3))

    The when a microstructure is discretized, the different local states are
    mapped into local state space, which results in an array of shape
    `(n_samples, n_x, n_y, n_states)`, where `n_states=3` in this case.
    For example, if a cell has a label of 2, its local state will be
    `[0, 0, 1]`. The local state can only have values of 0 or 1.

    >>> prim_basis = PrimitiveBasis(n_states=3)
    >>> X_prim = np.array([[[[0, 1, 0],
    ...                      [0, 1, 0],
    ...                      [1, 0, 0]],
    ...                     [[0, 1, 0],
    ...                      [1, 0, 0],
    ...                      [0, 0, 1]],
    ...                     [[1, 0, 0],
    ...                      [0, 1, 0],
    ...                      [1, 0, 0]]]])
    >>> assert(np.allclose(X_prim, prim_basis.discretize(X)))

    Check that the basis works when all the states are present in the
    microstructure.

    >>> prim_basis = PrimitiveBasis(n_states=3)
    >>> X = np.array([1, 1, 0])
    >>> X_prim = np.array([[0, 1, 0],
    ...                    [0, 1, 0],
    ...                    [1, 0, 0]])
    >>> assert(np.allclose(X_prim, prim_basis.discretize(X)))

    In previous two microstructures had values that fell on the peak of the
    primitive (or hat) basis functions. If a local state value falls between
    two peaks of the primitive basis functions the value will be shared by both
    basis functions. To ensure that all local states fall between the peaks
    of two basis functions, we need to specify the local state domain. For
    example, if a cell has a value of 0.4, and the basis has n_states=2 and
    the domain=[0, 1] then the local state is (0.6, 0.4) (the local state
    must sum to 1).

    Here are a few examples where the local states fall between the picks of
    local states. The first specifies the local state space domain between
    `[0, 1]`.

    >>> n_states = 10
    >>> np.random.seed(4)
    >>> X = np.random.random((2, 5, 3, 2))
    >>> X_ = PrimitiveBasis(n_states, [0, 1]).discretize(X)
    >>> H = np.linspace(0, 1, n_states)
    >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
    >>> assert np.allclose(X, Xtest)

    Here is an example where the local state space domain is between `[-1, 1]`.

    >>> n_states = 3
    >>> X = np.array([-1, 0, 1, 0.5])
    >>> X_test = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]]
    >>> X_ = PrimitiveBasis(n_states, [-1, 1]).discretize(X)
    >>> assert np.allclose(X_, X_test)

    If the local state values in the microstructure are outside of the domain
    they can no longer be represented by two primitive basis functions and
    violates constraint above.

    >>> n_states = 2
    >>> X = np.array([-1, 1])
    >>> prim_basis = PrimitiveBasis(n_states, domain=[0, 1])
    >>> prim_basis.discretize(X)
    Traceback (most recent call last):
    ...
    RuntimeError: X must be within the specified domain

    """

    def _select_slice(self, ijk, s0):
        """
        Helper method used to calibrate influence coefficients from in
        mks_localization_model to account for redundancies from linearly
        dependent local states.
        """
        if np.all(np.array(ijk) == 0):
            s1 = s0
        else:
            s1 = (slice(-1),)
        return s1

    def discretize(self, X):
        """
        Discretize `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is thes patial discretization.
        Returns:
            Float valued field of local states between 0 and 1.
        """
        self.check(X)
        self._select_axes(X)
        H = np.linspace(self.domain[0], self.domain[1], max(self.n_states) + 1)
        X_ = np.maximum(1 - (abs(X[..., None] - H)) / (H[1] - H[0]), 0)
        return X_[..., list(self.n_states)]
