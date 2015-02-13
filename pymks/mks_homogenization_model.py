from pymks.stats import correlate
import numpy as np


class MKSHomogenizationModel(object):

    '''
    The `MKSHomogenizationModel` takes in microstructures and a their
    associated macroscopic property, and created a low dimensional structure
    property linkage. The `MKSHomogenizationModel` model is designed to
    integrate with dimensionality reduction techniques and predictive models.

    Below is an examlpe of using MKSHomogenizationModel to predict the type of
    microstructure using PCA and Logistic Regression.

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
    >>> assert np.allclose(y0_test, [0, 0, 0])
    >>> assert np.allclose(y1_test, [1, 1, 1])

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
        if self.reducer is None:
            raise RuntimeError("dimension_reducer not specified")
        if self.linker is None:
            raise RuntimeError("property_linker not specified.")
        if not callable(getattr(self.reducer, "fit_transform", None)):
            raise RuntimeError(
                "dimension_reducer does not have fit_transform() method.")
        if not callable(getattr(self.reducer, "transform", None)):
            raise RuntimeError(
                "dimension_reducer does not have transform() method.")
        if not callable(getattr(self.linker, "fit", None)):
            raise RuntimeError(
                "property_linker does not have fit() method.")
        if not callable(getattr(self.linker, "predict", None)):
            raise RuntimeError(
                "property_linker does not have predict() method.")

    def fit(self, X, y, reducer_label=None):
        '''
        Fits data by calculating 2-point statistics from X, preforming
        dimension reduction using dimension_reducer, and fitting the reduced
        data with the property_linker.

        >>> from sklearn.decomposition import PCA
        >>> from sklearn.linear_model import LinearRegression
        >>> from pymks.bases import DiscreteIndicatorBasis
        >>> from pymks.stats import correlate

        >>> reducer = PCA(n_components=2)
        >>> linker = LinearRegression()
        >>> dbasis = DiscreteIndicatorBasis(n_states=2, domain=[0, 1])
        >>> model = MKSHomogenizationModel(dbasis,
        ...                                    dimension_reducer=reducer,
        ...                                    property_linker=linker)
        >>> np.random.seed(99)
        >>> X = np.random.randint(2, size=(3, 15))
        >>> y = np.array([1, 2, 3])
        >>> model.fit(X, y)
        >>> X_ = dbasis.discretize(X)
        >>> X_corr = correlate(X_)

        >>> X_pca = reducer.fit_transform(X_corr.reshape((X_corr.shape[0],
        ...                                               X_corr[0].size)))
        >>> assert np.allclose(model.data, X_pca)

        Args:
          X: The microstructure, an `(S, N, ...)` shaped
             array where `S` is the number of samples and `N` is the
             spatial discretization.
          y: The material property associated with `X`.
          reducer_label: label for X used during the fit_transform method
             for the `dimension_reducer`.
        '''
        X_preped = self._X_prep(X)
        if reducer_label is not None:
            X_reduced = self.reducer.fit_transform(X_preped, reducer_label)
        else:
            X_reduced = self.reducer.fit_transform(X_preped)
        self.linker.fit(X_reduced, y)
        self.data = X_reduced

    def predict(self, X):
        '''Predicts macroscopic property for the microstructures `X`.

        >>> from sklearn.manifold import LocallyLinearEmbedding
        >>> from sklearn.linear_model import BayesianRidge
        >>> from pymks.bases import DiscreteIndicatorBasis
        >>> np.random.seed(99)
        >>> X = np.random.randint(2, size=(50, 100))
        >>> y = np.random.random(50)
        >>> reducer = LocallyLinearEmbedding()
        >>> linker = BayesianRidge()
        >>> basis = DiscreteIndicatorBasis(2, domain=[0, 1])
        >>> model = MKSHomogenizationModel(basis,
        ...                                dimension_reducer=reducer,
        ...                                property_linker=linker)
        >>> model.fit(X, y)
        >>> X_test = np.random.randint(2, size=(1, 100))
        >>> assert np.allclose(model.predict(X_test), 0.53036321)


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

        >>> from sklearn.manifold import Isomap
        >>> from sklearn.linear_model import ARDRegression
        >>> from pymks.bases import DiscreteIndicatorBasis
        >>> reducer = Isomap()
        >>> linker = ARDRegression()
        >>> basis = DiscreteIndicatorBasis(2, [0, 1])
        >>> model = MKSHomogenizationModel(basis, reducer, linker)
        >>> X = np.array([[0, 1],
        ...               [1, 0]])
        >>> X_prep = model._X_prep(X)
        >>> X_test = np.array([[0, 0, 0, 0, 0.5, 0.5, 0, 0],
        ...                    [0, 0, 1, 1, 0.5, 0.5, 0, 0]])
        >>> assert np.allclose(X_test, X_prep)


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
