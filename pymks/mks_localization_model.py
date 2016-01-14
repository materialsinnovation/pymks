import numpy as np
from sklearn.linear_model import LinearRegression
from .filter import Filter
from .filter import _import_pyfftw
from scipy.linalg import lstsq
_import_pyfftw()


class MKSLocalizationModel(LinearRegression):

    """
    The `MKSLocalizationModel` fits data using the Materials Knowledge
    System in Fourier Space. The following demonstrates the viability
    of the `MKSLocalizationModel` with a simple 1D filter.

    Attributes:
        basis: Basis function used to discretize the microstucture.
        n_states: Interger value for number of local states, if a basis
            is specified, n_states indicates the order of the polynomial.
        coef_: Array of values that are the influence coefficients.

    >>> n_states = 2
    >>> n_spaces = 81
    >>> n_samples = 400

    Define a filter function.

    >>> def filter(x):
    ...     return np.where(x < 10,
    ...                     np.exp(-abs(x)) * np.cos(x * np.pi),
    ...                     np.exp(-abs(x - 20)) * np.cos((x - 20) * np.pi))

    Use the filter function to construct some coefficients.

    >>> coef_ = np.linspace(1, 0, n_states)[None,:] * filter(np.linspace(0, 20,
    ...                                                      n_spaces))[:,None]
    >>> Fcoef_ = np.fft.fft(coef_, axis=0)

    Make some test samples.

    >>> np.random.seed(2)
    >>> X = np.random.random((n_samples, n_spaces))

    Construct a response with the `Fcoef_`.

    >>> H = np.linspace(0, 1, n_states)
    >>> X_ = np.maximum(1 - abs(X[:,:,None] - H) / (H[1] - H[0]), 0)
    >>> FX = np.fft.fft(X_, axis=1)
    >>> Fy = np.sum(Fcoef_[None] * FX, axis=-1)
    >>> y = np.fft.ifft(Fy, axis=1).real

    Use the `MKSLocalizationModel` to reconstruct the coefficients

    >>> from .bases import PrimitiveBasis
    >>> prim_basis = PrimitiveBasis(n_states, [0, 1])
    >>> model = MKSLocalizationModel(basis=prim_basis)
    >>> model.fit(X, y)

    Check the result

    >>> assert np.allclose(np.fft.fftshift(coef_, axes=(0,)), model.coef_)
    """

    def __init__(self, basis, n_states=None, lstsq_rcond=None):
        """
        Instantiate a MKSLocalizationModel.

        Args:
            basis (class): an instance of a bases class.
            n_states (int, optional): number of local states
            lstsq_rcond (float, optional): rcond argument to linalg.lstsq
            function. Defaults to 4 orders of magnitude above machine
            epsilon.

        """
        self.basis = basis
        self.n_states = n_states
        if n_states is None:
            self.n_states = basis.n_states
        self.domain = basis.domain
        #any singular values not 4 orders of magnitude above machine epsilon
        #are considered linearly dependent and discarded
        self.lstsq_rcond = lstsq_rcond
        if self.lstsq_rcond is None:
            self.lstsq_rcond = np.finfo(float).eps*1e4

    def fit(self, X, y, size=None):
        """
        Fits the data by calculating a set of influence coefficients.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
            y (ND array): The response field, same shape as `X`.
            size (tuple, optional): Alters the shape of X and y during the
                calibration of the influence coefficients. If None, the size
                of the influence coefficients is the same shape as `X` and `y`.

        Example

        >>> X = np.linspace(0, 1, 4).reshape((1, 2, 2))
        >>> y = X.swapaxes(1, 2)
        >>> from .bases import PrimitiveBasis
        >>> prim_basis = PrimitiveBasis(2, [0, 1])
        >>> model = MKSLocalizationModel(basis=prim_basis)
        >>> model.fit(X, y)
        >>> assert np.allclose(model._filter.Fkernel, [[[ 0.5,  0.5],
        ...                                             [  -2,    0]],
        ...                                            [[-0.5,  0  ],
        ...                                             [  -1,  0  ]]])
        """
        self.basis = self.basis.__class__(self.n_states, self.domain)
        if size is not None:
            y = self.basis._reshape_feature(y, size)
            X = self.basis._reshape_feature(X, size)

        # if not len(y.shape) > 1:
        #     raise RuntimeError("The shape of y is incorrect.")
        # if y.shape != X.shape:
        #     raise RuntimeError("X and y must be the same shape.")
        self.basis._shape_check(X, y)  # call error check for shapes of X and y

        X_ = self.basis.discretize(X)
        axes = np.arange(X_.ndim)[1:-1]
        FX = np.fft.fftn(X_, axes=axes)
        Fy = np.fft.fftn(y, axes=axes)
        Fkernel = np.zeros(FX.shape[1:], dtype=np.complex)
        s0 = (slice(None),)
        for ijk in np.ndindex(X_.shape[1:-1]):
            s1 = self.basis._select_slice(ijk, s0)
            Fkernel[ijk + s1] = lstsq(FX[s0 + ijk + s1], Fy[s0 + ijk], self.lstsq_rcond)[0]

        self._filter = Filter(Fkernel[None])

    @property
    def coef_(self):
        """Returns the coefficients in real space with origin shifted to the
        center.
        """
        return self._filter._frequency_2_real()[0]

    def predict(self, X):
        """Predicts a new response from the microstructure function `X` with
        calibrated influence coefficients.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
        Returns:
            The predicted response field the same shape as `X`.

        Example

        >>> X = np.linspace(0, 1, 4).reshape((1, 2, 2))
        >>> y = X.swapaxes(1, 2)
        >>> from .bases import PrimitiveBasis
        >>> prim_basis = PrimitiveBasis(2, [0, 1])
        >>> model = MKSLocalizationModel(basis=prim_basis)
        >>> model.fit(X, y)
        >>> assert np.allclose(y, model.predict(X))

        The fit method must be called to calibrate the coefficients before
        the predict method can be used.

        >>> MKSModel = MKSLocalizationModel(prim_basis)
        >>> MKSModel.predict(X)
        Traceback (most recent call last):
        ...
        AttributeError: fit() method must be run before predict().
        """

        if not hasattr(self, '_filter'):
            raise AttributeError("fit() method must be run before predict().")
        y_pred_shape = self.basis._output_shape(X)
        X = self.basis._reshape_feature(X, self._filter.Fkernel.shape[1:-1])
        X_ = self.basis.discretize(X)
        return self._filter.convolve(X_).reshape(y_pred_shape)

    def resize_coeff(self, size):
        """Scale the size of the coefficients and pad with zeros.

        Args:
            size (tuple): The new size of the influence coefficients.

        Returns:
            The resized influence coefficients to size.

        Example

        Let's first instantitate a model and fabricate some
        coefficients.

        >>> from pymks.bases import PrimitiveBasis
        >>> prim_basis = PrimitiveBasis(n_states=2)
        >>> model = MKSLocalizationModel(prim_basis)
        >>> coef_ = np.arange(20).reshape((5, 4, 1))
        >>> coef_ = np.concatenate((coef_ , np.ones_like(coef_)), axis=2)
        >>> coef_ = np.fft.ifftshift(coef_, axes=(0, 1))
        >>> model._filter = Filter(np.fft.fftn(coef_, axes=(0, 1))[None])

        The coefficients can be reshaped by passing the new shape that
        coefficients should have.

        >>> model.resize_coeff((10, 7))
        >>> assert np.allclose(model.coef_[:,:,0],
        ...                    [[0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 1, 2, 3, 0],
        ...                     [0, 0, 4, 5, 6, 7, 0],
        ...                     [0, 0, 8, 9,10,11, 0],
        ...                     [0, 0,12,13,14,15, 0],
        ...                     [0, 0,16,17,18,19, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0],
        ...                     [0, 0, 0, 0, 0, 0, 0]])
        """
        self._filter.resize(size)

    def _test(self):
        """Tests

        >>> n_states = 10
        >>> np.random.seed(3)
        >>> X = np.random.random((2, 5, 3))
        >>> from .bases import PrimitiveBasis
        >>> prim_basis = PrimitiveBasis(n_states, [0, 1])
        >>> X_ = prim_basis.discretize(X)
        >>> H = np.linspace(0, 1, n_states)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        Use Legendre polynomials basis for the microstructure function
        and take the Fourier transform.

        >>> from .bases import LegendreBasis
        >>> np.random.seed(3)
        >>> X = np.random.random((1, 3, 3))
        >>> leg_basis = LegendreBasis(2, [0, 1])
        >>> model = MKSLocalizationModel(basis=leg_basis)
        >>> X_ = leg_basis.discretize(X)
        >>> FX = np.fft.fftn(X_, axes=(1, 2))
        >>>
        >>> FXtest = np.array([[[[4.50000000+0.j, -0.79735949+0.],
        ...                      [0.00000000+0.j, -1.00887157-1.48005289j],
        ...                      [0.00000000+0.j, -1.00887157+1.48005289j]],
        ...                     [[0.00000000+0.j, 0.62300683-4.97732233j],
        ...                      [0.00000000+0.j, 1.09318216+0.10131035j],
        ...                      [0.00000000+0.j, 0.37713401+1.87334545j]],
        ...                     [[0.00000000+0.j, 0.62300683+4.97732233j],
        ...                      [0.00000000+0.j, 0.37713401-1.87334545j],
        ...                      [0.00000000+0.j, 1.09318216-0.10131035j]]]])
        >>> assert np.allclose(FX, FXtest)
        """
        pass
