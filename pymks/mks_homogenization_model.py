from pymks.stats import correlate
from sklearn.base import BaseEstimator
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np


class MKSHomogenizationModel(BaseEstimator):

    """
    The `MKSHomogenizationModel` takes in microstructures and a their
    associated macroscopic property, and created a low dimensional structure
    property linkage. The `MKSHomogenizationModel` model is designed to
    integrate with dimensionality reduction techniques and predictive models.

    Attributes:
        degree: Degree of the polynomial used by
            `property_linker`.
        n_components: Number of components used by `dimension_reducer`.
        dimension_reducer: Instance of a dimensionality reduction class.
        property_linker: Instance of class that maps materials property to the
            microstuctures.
        correlations: spatial correlations to be computed
        basis: instance of a basis class

    Below is an examlpe of using MKSHomogenizationModel to predict (or
    classify) the type of microstructure using PCA and Logistic Regression.

    >>> n_states = 3
    >>> domain = [-1, 1]

    >>> from pymks.bases import LegendreBasis
    >>> leg_basis = LegendreBasis(n_states=n_states, domain=domain)
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.linear_model import LogisticRegression
    >>> reducer = PCA(n_components=3)
    >>> linker = LogisticRegression()
    >>> model = MKSHomogenizationModel(
    ...     basis=leg_basis, dimension_reducer=reducer, property_linker=linker)
    >>> from pymks.datasets import make_cahn_hilliard
    >>> X0, X1 = make_cahn_hilliard(n_samples=50)
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
    """

    def __init__(self, basis=None, dimension_reducer=None, n_components=None,
                 property_linker=None, degree=1, correlations=None):
        """
        Create an instance of a `MKSHomogenizationModel`.

        Args:
            basis (class, optional): an instance of a bases class.
            dimension_reducer (class, optional): an instance of a
                dimensionality reduction class with a fit_transform method.
            property_linker (class, optional): an instance for a machine
                learning class with fit and predict methods.
            n_components (int, optional): number of components kept by the
                dimension_reducer
            degree (int, optional): degree of the polynomial used by
                property_linker.
            correlations (list, optional): list of spatial correlations to
                compute, default is the autocorrelation with the first local
                state and all of its cross correlations. For example if basis
                has n_states=3, correlation would be [(0, 0), (0, 1), (0, 2)]
        """

        self.basis = basis
        self.dimension_reducer = dimension_reducer
        if self.dimension_reducer is None:
            self.dimension_reducer = RandomizedPCA()
        if n_components is None:
            n_components = self.dimension_reducer.n_components
        if n_components is None:
            n_components = 2
        if property_linker is None:
            property_linker = LinearRegression()
        if correlations is None and basis is not None:
            correlations = [(0, l) for l in range(basis.n_states)]
        self._linker = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                                ('connector', property_linker)])
        self._check_methods
        self.degree = degree
        self.n_components = n_components
        self.property_linker = property_linker
        self.correlations = correlations
        self._fit = False

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        """Setter for the number of components using by the dimension_reducer
        """
        self._n_components = value
        self.dimension_reducer.n_components = value

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        """Setter for the polynomial degree for property_linker.
        """
        self._degree = value
        self._linker.set_params(poly__degree=value)

    @property
    def property_linker(self):
        return self._property_linker

    @property_linker.setter
    def property_linker(self, prop_linker):
        """Setter for the property_linker class.
        """
        self._property_linker = prop_linker
        self._linker.set_params(connector=prop_linker)

    def _check_methods(self):
        """
        Helper function to make check that the dimensionality reduction and
        property linking methods have the appropriate methods.
        """
        if not callable(getattr(self.dimension_reducer,
                                "fit_transform", None)):
            raise RuntimeError(
                "dimension_reducer does not have fit_transform() method.")
        if not callable(getattr(self.dimension_reducer, "transform", None)):
            raise RuntimeError(
                "dimension_reducer does not have transform() method.")
        if not callable(getattr(self.linker, "fit", None)):
            raise RuntimeError(
                "property_linker does not have fit() method.")
        if not callable(getattr(self.linker, "predict", None)):
            raise RuntimeError(
                "property_linker does not have predict() method.")

    def fit(self, X, y, reduce_labels=None,
            periodic_axes=None, confidence_index=None, size=None):
        """
        Fits data by calculating 2-point statistics from X, preforming
        dimension reduction using dimension_reducer, and fitting the reduced
        data with the property_linker.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
            y (1D array): The material property associated with `X`.
            reducer_labels (1D array, optional): label for X used during the
                fit_transform method for the `dimension_reducer`.
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Example

        >>> from sklearn.decomposition import PCA
        >>> from sklearn.linear_model import LinearRegression
        >>> from pymks.bases import PrimitiveBasis
        >>> from pymks.stats import correlate

        >>> reducer = PCA(n_components=2)
        >>> linker = LinearRegression()
        >>> prim_basis = PrimitiveBasis(n_states=2, domain=[0, 1])
        >>> correlations = [(0, 0), (1, 1), (0, 1)]
        >>> model = MKSHomogenizationModel(prim_basis,
        ...                                dimension_reducer=reducer,
        ...                                property_linker=linker,
        ...                                correlations=correlations)
        >>> np.random.seed(99)
        >>> X = np.random.randint(2, size=(3, 15))
        >>> y = np.array([1, 2, 3])
        >>> model.fit(X, y)
        >>> X_ = prim_basis.discretize(X)
        >>> X_stats = correlate(X_)
        >>> X_reshaped = X_stats.reshape((X_stats.shape[0], X_stats[0].size))
        >>> X_pca = reducer.fit_transform(X_reshaped - np.mean(X_reshaped,
        ...                               axis=1)[:, None])
        >>> assert np.allclose(model.fit_data, X_pca)
        """
        if periodic_axes is None:
            periodic_axes = []
        if size is not None:
            new_shape = (X.shape[0],) + size
            X = X.reshape(new_shape)
        X_stats = self._correlate(X, periodic_axes, confidence_index)
        self.stats_fit(X_stats, y, reduce_labels)

    def predict(self, X, periodic_axes=None, confidence_index=None):
        """Predicts macroscopic property for the microstructures `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Returns:
            The predicted macroscopic property for `X`.

        Example

        >>> from sklearn.manifold import LocallyLinearEmbedding
        >>> from sklearn.linear_model import BayesianRidge
        >>> from pymks.bases import PrimitiveBasis
        >>> np.random.seed(99)
        >>> X = np.random.randint(2, size=(50, 100))
        >>> y = np.random.random(50)
        >>> reducer = LocallyLinearEmbedding()
        >>> linker = BayesianRidge()
        >>> prim_basis = PrimitiveBasis(2, domain=[0, 1])
        >>> model = MKSHomogenizationModel(prim_basis, n_components=2,
        ...                                dimension_reducer=reducer,
        ...                                property_linker=linker)
        >>> model.fit(X, y)
        >>> X_test = np.random.randint(2, size=(1, 100))
        >>> assert np.allclose(model.predict(X_test), 0.53042314)
        """
        if periodic_axes is None:
            periodic_axes = []
        if not self._fit:
            raise RuntimeError('fit() method must be run before predict().')
        X_stats = self._correlate(X, periodic_axes, confidence_index)
        return self.stats_predict(X_stats)

    def stats_fit(self, X_stats, y, reduce_labels=None):
        """Fit model using spatial correlations

        Args:
            X_stats (ND array): spatial correlations with dimensions (n_states,
                n_x, ..., n_correlations) where `n_samples` is the number of
                samples, `n_x` is the spatial discretization and
                `n_correlations is the spatial correlations (2-point
                statistics).
            y (1D array): The material property associated with `X_stats`.
            reducer_labels (1D array, optional): label for X used during the
                fit_transform method for the `dimension_reducer`.

        Example

        >>> from sklearn.decomposition import PCA
        >>> from sklearn.linear_model import LinearRegression
        >>> from pymks.bases import PrimitiveBasis
        >>> from pymks.stats import correlate

        >>> reducer = PCA(n_components=2)
        >>> linker = LinearRegression()
        >>> prim_basis = PrimitiveBasis(n_states=2, domain=[0, 1])
        >>> correlations = [(0, 0), (1, 1), (0, 1)]
        >>> model = MKSHomogenizationModel(dimension_reducer=reducer,
        ...                                property_linker=linker)
        >>> np.random.seed(99)
        >>> X = np.random.randint(2, size=(3, 15))
        >>> y = np.array([1, 2, 3])
        >>> X_ = prim_basis.discretize(X)
        >>> X_stats = correlate(X_, correlations=correlations)
        >>> model.stats_fit(X_stats, y)
        >>> X_reshaped = X_stats.reshape((X_stats.shape[0], X_stats[0].size))
        >>> X_pca = reducer.fit_transform(X_reshaped - np.mean(X_reshaped,
        ...                               axis=1)[:, None])
        >>> assert np.allclose(model.fit_data, X_pca)

        """
        X_reshape = self._reshape(X_stats)
        X_reduced = self.dimension_reducer.fit_transform(X_reshape,
                                                         reduce_labels)
        self._linker.fit(X_reduced, y)
        self.fit_data = X_reduced
        self._fit = True

    def stats_predict(self, X_stats):
        """Predicts macroscopic property for 2-point statistics `X_stats`.

        Args:
            X_stats (ND array): spatial correlations with dimensions (n_states,
                n_x, ..., n_correlations) where `n_samples` is the number of
                samples, `n_x` is the spatial discretization and
                `n_correlations is the spatial correlations (2-point
                statistics).

        Returns:
            The predicted macroscopic property for `X_stats`.

        Example

        >>> from sklearn.manifold import LocallyLinearEmbedding
        >>> from sklearn.linear_model import BayesianRidge
        >>> from pymks.bases import PrimitiveBasis
        >>> from pymks.stats import correlate
        >>> np.random.seed(99)
        >>> X = np.random.randint(2, size=(50, 100))
        >>> y = np.random.random(50)
        >>> reducer = LocallyLinearEmbedding()
        >>> linker = BayesianRidge()
        >>> prim_basis = PrimitiveBasis(2, domain=[0, 1])
        >>> model = MKSHomogenizationModel(basis=prim_basis, n_components=2,
        ...                                dimension_reducer=reducer,
        ...                                property_linker=linker)
        >>> X_ = prim_basis.discretize(X)
        >>> model.fit(X, y)

        Predict with `stats_predict`

        >>> correlations = [(0, 0), (0, 1)]
        >>> X_test = np.random.randint(2, size=(1, 100))
        >>> X_test_ = prim_basis.discretize(X_test)
        >>> X_test_stats = correlate(X_test_, correlations=correlations)
        >>> assert np.allclose(model.stats_predict(X_test_stats), 0.53042314)
        """
        X_reshape = self._reshape(X_stats)
        X_reduced = self.dimension_reducer.transform(X_reshape)
        self.predict_data = X_reduced
        return self._linker.predict(X_reduced)

    def _correlate(self, X, periodic_axes, confidence_index):
        """
        Helper function used to calculated 2-point statistics from `X` and
        reshape them appropriately for fit and predict methods.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization..
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Returns:
            Spatial correlations for each sample formated with dimensions
            (n_samples, n_features).

        Example

        >>> from sklearn.manifold import Isomap
        >>> from sklearn.linear_model import ARDRegression
        >>> from pymks.bases import PrimitiveBasis
        >>> reducer = Isomap()
        >>> linker = ARDRegression()
        >>> prim_basis = PrimitiveBasis(2, [0, 1])
        >>> model = MKSHomogenizationModel(prim_basis, reducer, linker)
        >>> X = np.array([[0, 1],
        ...               [1, 0]])
        >>> X_stats = model._correlate(X, [], None)
        >>> X_test = np.array([[[ 0, 0],
        ...                     [0.5, 0]],
        ...                    [[0, 1,],
        ...                     [0.5, 0]]])
        >>> assert np.allclose(X_test, X_stats)
        """
        if self.basis is None:
            raise AttributeError('basis must be specified')
        X_ = self.basis.discretize(X)
        X_stats = correlate(X_, periodic_axes=periodic_axes,
                            confidence_index=confidence_index,
                            correlations=self.correlations)
        return X_stats

    def _reshape(self, X_stats):
        """
        Helper function used to reshape 2-point statistics appropriately for
        fit and predict methods.

        Args:
            `X_stats`: The discretized microstructure function, an
                `(n_samples, n_x, ..., n_states)` shaped array
                Where `n_samples` is the number of samples, `n_x` is thes
                patial discretization, and n_states is the number of local
                states.

        Returns:
            Spatial correlations for each sample formated with dimensions
            (n_samples, n_features).

        Example
        >>> X_stats = np.zeros((2, 2, 2, 2))
        >>> X_stats[1] = 3.
        >>> X_stats[..., 1] = 1.
        >>> X_results = np.array([[-.5, .5, -.5, .5, -.5, .5, -.5,  0.5],
        ...                       [1., -1., 1., -1., 1., -1., 1., -1.]])
        >>> from pymks import PrimitiveBasis
        >>> prim_basis = PrimitiveBasis(2)
        >>> model = MKSHomogenizationModel(prim_basis)
        >>> assert np.allclose(X_results, model._reshape(X_stats))
        """
        X_reshaped = X_stats.reshape((X_stats.shape[0], X_stats[0].size))
        return X_reshaped - np.mean(X_reshaped, axis=1)[:, None]

    def score(self, X, y, periodic_axes=None, confidence_index=None):
        """
        The score function for the MKSHomogenizationModel. It formats the
        data and uses the score method from the property_linker.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
            y (1D array): The material property associated with `X`.
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Returns:
             Score for MKSHomogenizationModel from the selected
             property_linker.
        """
        if periodic_axes is None:
            periodic_axes = []
        if not callable(getattr(self._linker, "score", None)):
            raise RuntimeError(
                "property_linker does not have score() method.")
        X_corr = self._correlate(X, periodic_axes, confidence_index)
        X_reshaped = self._reshape(X_corr)
        X_reduced = self.dimension_reducer.transform(X_reshaped)
        return self._linker.score(X_reduced, y)
