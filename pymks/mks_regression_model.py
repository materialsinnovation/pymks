import numpy as np
from sklearn.linear_model import LinearRegression
from .filter import Filter


class MKSRegressionModel(LinearRegression):
    '''
    The `MKSRegressionModel` fits data using the Materials Knowledge
    System in Fourier Space. The following demonstrates the viability
    of the `MKSRegressionModel` with a simple 1D filter.

    >>> n_states = 2
    >>> n_spaces = 81
    >>> n_samples = 400

    Define a filter function.

    >>> def filter(x):
    ...     return np.where(x < 10,
    ...                     np.exp(-abs(x)) * np.cos(x * np.pi),
    ...                     np.exp(-abs(x - 20)) * np.cos((x - 20) * np.pi))

    Use the filter function to construct some coefficients.

    >>> coeff = np.linspace(1, 0, n_states)[None,:] * filter(np.linspace(0, 20,
    ...                                                      n_spaces))[:,None]
    >>> Fcoeff = np.fft.fft(coeff, axis=0)

    Make some test samples.

    >>> np.random.seed(2)
    >>> X = np.random.random((n_samples, n_spaces))

    Construct a response with the `Fcoeff`.

    >>> H = np.linspace(0, 1, n_states)
    >>> X_ = np.maximum(1 - abs(X[:,:,None] - H) / (H[1] - H[0]), 0)
    >>> FX = np.fft.fft(X_, axis=1)
    >>> Fy = np.sum(Fcoeff[None] * FX, axis=-1)
    >>> y = np.fft.ifft(Fy, axis=1).real

    Use the `MKSRegressionModel` to reconstruct the coefficients

    >>> from .bases import ContinuousIndicatorBasis
    >>> basis = ContinuousIndicatorBasis(n_states, [0, 1])
    >>> model = MKSRegressionModel(basis=basis)
    >>> model.fit(X, y)

    Check the result

    >>> assert np.allclose(np.fft.fftshift(coeff, axes=(0,)), model.coeff)

    Attributes:
        n_states: Interger value for number of local states, if a basis is
          specified, n_states indicates the order of the polynomial.
        coef: Array of values that are the influence coefficients
        Fcoef: Frequency space representation of coef
    '''

    def __init__(self, basis, n_states=None):
        """
        Instantiate a MKSRegressionModel.

        Args:
          basis: an instance of a bases class.

        """
        self.basis = basis
        if n_states is None:
            self.n_states = basis.n_states
        else:
            self.n_states = n_states
        self.domain = basis.domain

    def fit(self, X, y):
        '''
        Fits the data by calculating a set of influence coefficients.

        >>> X = np.linspace(0, 1, 4).reshape((1, 2, 2))
        >>> y = X.swapaxes(1, 2)
        >>> from .bases import ContinuousIndicatorBasis
        >>> basis = ContinuousIndicatorBasis(2, [0, 1])
        >>> model = MKSRegressionModel(basis=basis)
        >>> model.fit(X, y)
        >>> assert np.allclose(model._filter.Fkernel, [[[ 0.5,  0.5],
        ...                                             [  -2,    0]],
        ...                                            [[-0.5,  0  ],
        ...                                             [  -1,  0  ]]])


        Args:
          X: the microstructure function, an `(S, N, ...)` shaped
             array where `S` is the number of samples and `N` is the
             spatial discretization.
          y: The response field, same shape as `X`.
        '''
        self.basis = self.basis.__class__(self.n_states, self.domain)

        if not len(y.shape) > 1:
            raise RuntimeError("The shape of y is incorrect.")
        if y.shape != X.shape:
            raise RuntimeError("X and y must be the same shape.")
        X_ = self.basis.discretize(X)
        axes = np.arange(len(X.shape) - 1) + 1
        FX = np.fft.fftn(X_, axes=axes)
        Fy = np.fft.fftn(y, axes=axes)
        Fkernel = np.zeros(FX.shape[1:], dtype=np.complex)
        s0 = (slice(None),)
        for ijk in np.ndindex(X.shape[1:]):
            if np.all(np.array(ijk) == 0):
                s1 = s0
            else:
                s1 = (slice(-1),)
            Fkernel[ijk + s1] = np.linalg.lstsq(FX[s0 + ijk + s1],
                                                Fy[s0 + ijk])[0]

        self._filter = Filter(Fkernel[None])

    @property
    def coeff(self):
        '''Returns the coefficients in real space with origin shifted to the
        center.
        '''
        return self._filter._frequency_2_real()[0]

    def predict(self, X):
        r'''Calculate a new response from the microstructure function `X` with
        calibrated influence coefficients.

        >>> X = np.linspace(0, 1, 4).reshape((1, 2, 2))
        >>> y = X.swapaxes(1, 2)
        >>> from .bases import ContinuousIndicatorBasis
        >>> basis = ContinuousIndicatorBasis(2, [0, 1])
        >>> model = MKSRegressionModel(basis=basis)
        >>> model.fit(X, y)
        >>> assert np.allclose(y, model.predict(X))

        The fit method must be called to calibrate the coefficients before
        the predict method can be used.

        >>> MKSmodel = MKSRegressionModel(basis)
        >>> MKSmodel.predict(X)
        Traceback (most recent call last):
        ...
        AttributeError: fit() method must be run before predict().

        Args:
            X: The microstructre function, an `(S, N, ...)` shaped
                array where `S` is the number of samples and `N`
                is the spatial discretization.
        Returns:
            The predicted response field the same shape as `X`.
        '''

        if not hasattr(self, '_filter'):
            raise AttributeError("fit() method must be run before predict().")
        X_ = self.basis.discretize(X)
        return self._filter.convolve(X_)

    def resize_coeff(self, size):
        '''Scale the size of the coefficients and pad with zeros.

        Let's first instantitate a model and fabricate some
        coefficients.

        >>> from pymks.bases import DiscreteIndicatorBasis
        >>> basis = DiscreteIndicatorBasis(n_states=2)
        >>> model = MKSRegressionModel(basis)
        >>> coeff = np.arange(20).reshape((5, 4, 1))
        >>> coeff = np.concatenate((coeff , np.ones_like(coeff)), axis=2)
        >>> coeff = np.fft.ifftshift(coeff, axes=(0, 1))
        >>> model._filter = Filter(np.fft.fftn(coeff, axes=(0, 1))[None])

        The coefficients can be reshaped by passing the new shape that
        coefficients should have.

        >>> model.resize_coeff((10, 7))
        >>> assert np.allclose(model.coeff[:,:,0],
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

        Args:
            size: The new size of the influence coefficients.
        Returns:
            The resized influence coefficients to size.
        '''
        self._filter.resize(size)

    def _test(self):
        '''Tests

        >>> n_states = 10
        >>> np.random.seed(3)
        >>> X = np.random.random((2, 5, 3))
        >>> from .bases import ContinuousIndicatorBasis
        >>> basis = ContinuousIndicatorBasis(n_states, [0, 1])
        >>> X_ = basis.discretize(X)
        >>> H = np.linspace(0, 1, n_states)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        Use Legendre polynomials basis for the microstructure function
        and take the Fourier transform.

        >>> from .bases import LegendreBasis
        >>> np.random.seed(3)
        >>> X = np.random.random((1, 3, 3))
        >>> basis = LegendreBasis(2, [0, 1])
        >>> model = MKSRegressionModel(basis=basis)
        >>> #FX = model._discrtizefft(X)
        >>> X_ = basis.discretize(X)
        >>> FX = np.fft.fftn(X_, axes=(1, 2))
        >>> FXtest = np.array([[[[-0.79735949+0. ,  4.50000000+0.j],
        ...                      [-1.00887157-1.48005289j,  0.00000000+0.j],
        ...                      [-1.00887157+1.48005289j,  0.00000000+0.j]],
        ...                     [[ 0.62300683-4.97732233j,  0.00000000+0.j],
        ...                      [ 1.09318216+0.10131035j,  0.00000000+0.j],
        ...                      [ 0.37713401+1.87334545j,  0.00000000+0.j]],
        ...                     [[ 0.62300683+4.97732233j,  0.00000000+0.j],
        ...                      [ 0.37713401-1.87334545j,  0.00000000+0.j],
        ...                      [ 1.09318216-0.10131035j,  0.00000000+0.j]]]])
        >>> assert np.allclose(FX, FXtest)
        '''
        pass
