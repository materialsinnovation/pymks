from .mks_structure_analysis import MKSStructureAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np


class MKSHomogenizationModel(MKSStructureAnalysis):

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
        reduced_fit_data: Low dimensionality representation of spatial
            correlations used to fit the model.
        reduced_predict_data: Low dimensionality representation of spatial
            correlations predicted by the model.
        periodic_axes: axes that are periodic. (0, 2) would indicate that
            axes x and z are periodic in a 3D microstrucure.
        coef_: Array of values that are the coefficients.
        intercept_: Value that are the intercept

    Below is an example of using MKSHomogenizationModel to predict (or
    classify) the type of microstructure using PCA and Logistic Regression.

    >>> import numpy as np
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
                 property_linker=None, degree=1, periodic_axes=None,
                 correlations=None, compute_correlations=True, n_jobs=1,
                 store_correlations=False, mean_center=True):
        """
        Create an instance of a `MKSHomogenizationModel`.

        Args:
            basis (class, optional): an instance of a bases class.
            dimension_reducer (class, optional): an instance of a
                dimensionality reduction class with a fit_transform method. The
                default class is RandomizedPCA.
            property_linker (class, optional): an instance for a machine
                learning class with fit and predict methods.
            n_components (int, optional): number of components kept by the
                dimension_reducer
            degree (int, optional): degree of the polynomial used by
                property_linker.
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            correlations (list, optional): list of spatial correlations to
                compute, default is the autocorrelation with the first local
                state and all of its cross correlations. For example if basis
                has basis.n_states=3, correlation would be [(0, 0), (0, 1),
                (0, 2)]. If n_states=[0, 2, 4], the default correlations are
                [(0, 0), (0, 2), (0, 4)] corresponding to the autocorrelations
                for the 0th local state, and the cross correlations with the 0
                and 2 as well as 0 and 4.
            compute_correlations (boolean, optional): If false spatial
                correlations will not be calculated as part of the fit and
                predict methods. The spatial correlations can be passed as `X`
                to both methods, default is True.
            n_jobs (int, optional): number of parallel jobs to run. only used
                if pyfftw is install.
            store_correlations (boolean, optional): indicate if spatial
                correlations should be stored
            mean_center (boolean, optional): If true the data will be mean
                centered before dimensionality reduction is computed.
        """

        if property_linker is None:
            property_linker = LinearRegression()
        self._linker = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                                ('connector', property_linker)])
        self.degree = degree
        self.property_linker = property_linker
        if not callable(getattr(self.property_linker, "fit", None)):
            raise RuntimeError(
                "property_linker does not have fit() method.")
        if not callable(getattr(self.property_linker, "predict", None)):
            raise RuntimeError(
                "property_linker does not have predict() method.")
        self.compute_correlations = compute_correlations
        self.reduced_fit_data = None
        self.reduced_predict_data = None
        if self.compute_correlations:
            if basis is None:
                raise RuntimeError(('a basis is need to compute spatial ') +
                                   ('correlations'))
        super(MKSHomogenizationModel,
              self).__init__(store_correlations=store_correlations,
                             dimension_reducer=dimension_reducer,
                             correlations=correlations, n_jobs=n_jobs,
                             n_components=n_components, basis=basis,
                             mean_center=mean_center,
                             periodic_axes=periodic_axes)

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
    def coef_(self):
        return self._linker.named_steps['connector'].coef_

    @coef_.setter
    def coef_(self, coef):
        """Setter for the coefficients for property_linker.
        """
        self._linker.named_steps['connector'].coef_ = coef

    @property
    def intercept_(self):
        return self._linker.named_steps['connector'].intercept_

    @intercept_.setter
    def intercept_(self, intercept):
        """Setter for the intercept for property_linker.
        """
        self._linker.named_steps['connector'].intercept_ = intercept

    @property
    def property_linker(self):
        return self._property_linker

    @property_linker.setter
    def property_linker(self, prop_linker):
        """Setter for the property_linker class.
        """
        self._property_linker = prop_linker
        self._linker.set_params(connector=prop_linker)

    def fit(self, X, y, reduce_labels=None, confidence_index=None, size=None):
        """
        Fits data by calculating 2-point statistics from X, preforming
        dimension reduction using dimension_reducer, and fitting the reduced
        data with the property_linker.

        Args:
            X (ND array): The microstructures or spatial correlations, a
                `(n_samples, n_x, ...)` shaped array where `n_samples` is the
                number of samples and `n_x` is the spatial discretization.
            y (1D array): The material property associated with `X`.
            reducer_labels (1D array, optional): label for X used during the
                fit_transform method for the `dimension_reducer`.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Example

        Let's first start with using the microstructure and effective
        properties.

        >>> import numpy as np
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
        >>> X_stats = correlate(X, prim_basis)
        >>> X_reshaped = X_stats.reshape((X_stats.shape[0], -1))
        >>> X_pca = reducer.fit_transform(X_reshaped - np.mean(X_reshaped,
        ...                               axis=1)[:, None])
        >>> assert np.allclose(model.reduced_fit_data, X_pca)

        Now let's use the same method with spatial correlations instead of
        microtructures.

        >>> from sklearn.decomposition import PCA
        >>> from sklearn.linear_model import LinearRegression
        >>> from pymks.bases import PrimitiveBasis
        >>> from pymks.stats import correlate

        >>> reducer = PCA(n_components=2)
        >>> linker = LinearRegression()
        >>> prim_basis = PrimitiveBasis(n_states=2, domain=[0, 1])
        >>> correlations = [(0, 0), (1, 1), (0, 1)]
        >>> model = MKSHomogenizationModel(dimension_reducer=reducer,
        ...                                property_linker=linker,
        ...                                compute_correlations=False)
        >>> np.random.seed(99)
        >>> X = np.random.randint(2, size=(3, 15))
        >>> y = np.array([1, 2, 3])
        >>> X_stats = correlate(X, prim_basis, correlations=correlations)
        >>> model.fit(X_stats, y)
        >>> X_reshaped = X_stats.reshape((X_stats.shape[0], X_stats[0].size))
        >>> X_pca = reducer.fit_transform(X_reshaped - np.mean(X_reshaped,
        ...                               axis=1)[:, None])
        >>> assert np.allclose(model.reduced_fit_data, X_pca)


        """
        if self.compute_correlations:
            if size is not None:
                X = self.basis._reshape_feature(X, size)
            X = self._compute_stats(X, confidence_index)
        X_reshape = self._reduce_shape(X)
        X_reduced = self._fit_transform(X_reshape, reduce_labels)
        self._linker.fit(X_reduced, y)

    def predict(self, X, confidence_index=None):
        """Predicts macroscopic property for the microstructures `X`.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Returns:
            The predicted macroscopic property for `X`.

        Example

        >>> import numpy as np
        >>> from sklearn.manifold import LocallyLinearEmbedding
        >>> from sklearn.linear_model import BayesianRidge
        >>> from pymks.bases import PrimitiveBasis
        >>> np.random.seed(1)
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

        Predict with microstructures

        >>> y_pred = model.predict(X_test)

        Predict with spatial correlations

        >>> from pymks.stats import correlate
        >>> model.compute_correlations = False
        >>> X_corr = correlate(X, prim_basis, correlations=[(0, 0)])
        >>> model.fit(X_corr, y)
        >>> X_corr_test = correlate(X_test, prim_basis,
        ...                         correlations=[(0, 0)])
        >>> y_pred_stats = model.predict(X_corr_test)
        >>> assert np.allclose(y_pred_stats, y_pred, atol=1e-3)

        """
        if not hasattr(self._linker.get_params()['connector'], "coef_"):
            raise RuntimeError('fit() method must be run before predict().')
        _size = self._size_axes(self.basis)
        X = self.basis._reshape_feature(X, tuple(_size))
        if self.compute_correlations is True:
            X = self._compute_stats(X, confidence_index)
        X_reduced = self._transform(X)
        self.reduced_predict_data = X_reduced
        return self._linker.predict(X_reduced)

    def score(self, X, y, confidence_index=None):
        """
        The score function for the MKSHomogenizationModel. It formats the
        data and uses the score method from the property_linker.

        Args:
            X (ND array): The microstructure, an `(n_samples, n_x, ...)`
                shaped array where `n_samples` is the number of samples and
                `n_x` is the spatial discretization.
            y (1D array): The material property associated with `X`.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Returns:
             Score for MKSHomogenizationModel from the selected
             property_linker.
        """
        if not callable(getattr(self._linker, "score", None)):
            raise RuntimeError(
                "property_linker does not have score() method.")
        _size = self._size_axes(self.basis)
        X = self.basis._reshape_feature(X, _size)
        if self.compute_correlations:
            X = self._compute_stats(X, confidence_index)
        X_reduced = self._transform(X)
        return self._linker.score(X_reduced, y)

    def _size_axes(self, basis):
        """Helper function used to get the correct size of the axes when using
        for both periodic and non-periodic axes.
        """
        _size = self.basis._axes_shape
        if self.periodic_axes is None or len(self.periodic_axes) != len(_size):
            _axes = range(len(_size))
            if self.periodic_axes is not None:
                [_axes.remove(a) for a in self.periodic_axes]
            _size = np.ones(len(_size)) * _size
            _size[_axes] *= .5
        return tuple(_size)
