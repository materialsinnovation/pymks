from pymks.stats import correlate
import numpy as np


class MKSHomogenizationModel(object):

    '''
    The `MKSHomogenizationModel` takes in microstructures and a their
    associated macroscopic property, and created a low dimensional structure
    property linkage.

    The `MKSHomogenizationModel` model is designed to integrate with
    dimensionality reduction and techniques.

    In ord

    >>> n_states = 3
    >>> domain = [-1, 1]


    >>> from .bases import LegendreBasis
    >>> basis = LegendreBasis(n_states=n_states, domain=domain)
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.linear_model import LogisticRegression
    >>> reducer = PCA(n_components=3)
    >>> linker = LogisticRegression()
    >>> model = MKSHomogenizationModel(basis=basis, dimension_reducer=reducer,
    ...                                property_linker=linker)

    >>> from .datasets import make_cahn_hilliard
    >>> X0, X1 = make_cahn_hilliard(n_samples=25)
    >>> y0 = np.zeros(X0.shape[0])
    >>> y1 = np.ones(X1.shape[0])

    >>> X = np.concatenate((X0, X1))
    >>> y = np.concatenate((y0, y1))

    >>> model.fit(X, y)

    >>> X0_test, X1_test = make_cahn_hilliard(n_samples=3)
    >>> y0_test = model.predict(X0_test)
    >>> y1_test = model.predict(X1_test)
    >>> assert(np.allclose(y0_test, [0, 0, 0]))
    >>> assert(np.allclose(y1_test, [1, 1, 1]))

    '''

    def __init__(self, basis, dimension_reducer=None, property_linker=None):
        '''
        Create an instance of a `MKSHomogenizationModel`.
        Args:
            basis: an instance of a bases class.
            dimension_reducer: an instance of a dimensionality reduction
                class with a fit_transform method.
            property_linker: an instance for a machine learning class with fit
                and predict methods.
        '''

        self.basis = basis
        self.reducer = dimension_reducer
        self.linker = property_linker
        self.data = None
        if self.reducer is None:
            raise RuntimeError("dimension_reducer not specified")
        if self.linker is None:
            raise RuntimeError("property_linker not specified.")
        if not callable(getattr(self.reducer, "fit_transform", None)):
            raise RuntimeError(
                "dimension_reducer does not have fit_transform method.")
        if not callable(getattr(self.linker, "predict", None)):
            raise RuntimeError(
                "property_linker does not have predict method.")

    def fit(self, X, y, reducer_label=None):
        '''
        Fits data by calculating 2-point statistics from X, preforming
        dimension reduction using dimension_reducer, and fitting the reduced
        data with the property_linker.

        >>>

        Args:
          X: The microstructure, an `(S, N, ...)` shaped
             array where `S` is the number of samples and `N` is the
             spatial discretization.
          y: The material property associated with `X`.
          reducer_label: label for X used during the fit_transform method
             for the `dimension_reducer`.
        '''
        X_preped = self._X_prep(X)
        if y is not None:
            X_reduced = self.reducer.fit_transform(X_preped, y)
        else:
            X_reduced = self.reducer.fit_transform(X_preped)
        self.linker.fit(X_reduced, y)
        self.data = X_reduced

    def predict(self, X):
        '''Predicts macroscopic property for the microstructures `X`.

        >>>

        Args:
            X: The microstructre, an `(S, N, ...)` shaped array where `S` is
               the number of samples and `N` is the spatial discretization.
        Returns:
            The predicted macroscopic property for `X`.
        '''
        X_preped = self._X_prep(X)
        X_reduced = self.reducer.transform(X_preped)
        self.data = np.concatenate((self.data, X_reduced))
        return self.linker.predict(X_reduced)

    def _X_prep(self, X):
        '''
        Helper function used to calculated 2-point statistics from `X` and
        reshape them appropriately for fit and predict methods.

        >>>

        Args:
            X: The microstructre, an `(S, N, ...)` shaped array where `S` is
               the number of samples and `N` is the spatial discretization.
        Returns:
           Spatial correlations for each sample formated with dimensions
           (n_samples, n_features).
        '''
        X_ = self.basis.discretize(X)
        X_corr = correlate(X_)
        return X_corr.reshape((X_corr.shape[0], X_corr[0].size))
