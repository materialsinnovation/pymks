import numpy as np
from pymks.stats import correlate
from sklearn.base import BaseEstimator
from sklearn.decomposition import RandomizedPCA


class MKSStructureAnalysis(BaseEstimator):
    """
    `MKSStructureAnalysis` computes the 2-point statistics for a for a set of
    microstructures and does dimensionality reduction. It can be used to
    evaluate the selection of spatial correlations and look at clustering of
    2-point statistics.

    Attributes:
        n_components: Number of components used by `dimension_reducer`.
        dimension_reducer: Instance of a dimensionality reduction class.
        correlations: spatial correlations to be computed
        basis: instance of a basis class
        fit_data: Low dimensionality representation of spatial
            correlations used to fit the components.
        transformed_data: Reduced of spatial correlations.

    Below is an example of using MKSStructureAnalysis using FastICA.

    >>> from pymks.datasets import make_microstructure
    >>> from pymks.bases import PrimitiveBasis
    >>> from sklearn.decomposition import FastICA

    >>> leg_basis = PrimitiveBasis(n_states=2, domain=[0, 1])
    >>> reducer = FastICA(n_components=3)
    >>> analyzer = MKSStructureAnalysis(basis=leg_basis, mean_center=False,
    ...                                 dimension_reducer=reducer)

    >>> X = make_microstructure(n_samples=4, size=(13, 13), grain_size=(3, 3))
    >>> print(analyzer.fit_transform(X)) # doctest: +ELLIPSIS
    [[ 0.5 -0.5 -0.5]
     [ 0.5  0.5  0.5]
     [-0.5 -0.5  0.5]
     [-0.5  0.5 -0.5]]

    """

    def __init__(self, basis, correlations=None, dimension_reducer=None,
                 n_components=None, store_correlations=False,
                 mean_center=True):
        """
        Create an instance of a `MKSStructureAnalysis`.

        Args:
            basis: an instance of a bases class.
            dimension_reducer (class, optional): an instance of a
                dimensionality reduction class with a fit_transform method. The
                default class is RandomizedPCA.
            n_components (int, optional): number of components kept by the
                dimension_reducer
            correlations (list, optional): list of spatial correlations to
                compute, default is the autocorrelation with the first local
                state and all of its cross correlations. For example if basis
                has n_states=3, correlation would be [(0, 0), (0, 1), (0, 2)]
            store_correlations (boolean, optional): If true the computed
                2-point statistics will be saved as an attributes
                fit_correlations and transform_correlations.
            mean_center (boolean, optional): If true the data will be mean
                centered before dimensionality reduction is computed.
        """
        self.basis = basis
        self.correlations = correlations
        self.dimension_reducer = dimension_reducer
        self.store_correlations = store_correlations
        self.mean_center = mean_center
        if self.dimension_reducer is None:
            self.dimension_reducer = RandomizedPCA(copy=False)
        if n_components is None:
            n_components = self.dimension_reducer.n_components
        if n_components is None:
            n_components = 5
        self.n_components = n_components
        if self.correlations is None and basis is not None:
            self.correlations = [(0, l) for l in self.basis.n_states]
        if not callable(getattr(self.dimension_reducer,
                                "fit_transform", None)):
            raise RuntimeError(
                "dimension_reducer does not have fit_transform() method.")
        if not callable(getattr(self.dimension_reducer, "transform", None)):
            raise RuntimeError(
                "dimension_reducer does not have transform() method.")

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        """Setter for the number of components using by the dimension_reducer
        """
        self._n_components = value
        self.dimension_reducer.n_components = value

    def fit(self, X, reducer_labels=None, periodic_axes=None,
            confidence_index=None):
        """Fits data by using the 2-point statistics for X to fits the
        components used in dimensionality reduction.

        Args:
            X (ND array): The microstructures or spatial correlations, a
                `(n_samples, n_x, ...)` shaped array where `n_samples` is the
                number of samples and `n_x` is the spatial discretization.
            reducer_labels (1D array, optional): label for X used during the
                fit_transform method for the `dimension_reducer`.
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Example

        >>> from pymks.datasets import make_delta_microstructures
        >>> from pymks import PrimitiveBasis
        >>> n_states = 2
        >>> analyzer = MKSStructureAnalysis(basis=PrimitiveBasis(n_states),
        ...                              n_components=1)
        >>> np.random.seed(5)
        >>> size = (2, 3, 3)
        >>> X = np.random.randint(2, size=size)
        >>> analyzer.fit(X)
        >>> print(analyzer.dimension_reducer.components_.reshape(size)[0])
        ... # doctest: +ELLIPSIS
        [[ 0.02886463  0.02886463  0.02886463]
         [ 0.02886463 -0.43874233  0.49647159]
         [ 0.02886463  0.02886463 -0.17896069]]
        """
        X_stats = self._compute_stats(X, periodic_axes, confidence_index)
        self._fit_transform(X_stats, reducer_labels)

    def fit_transform(self, X, periodic_axes=None, confidence_index=None):
        """Fits data by using the 2-point statistics for X to fits the
        components used in dimensionality reduction and returns the reduction
        of the 2-point statistics for X.

        Args:
            X (ND array): The microstructures or spatial correlations, a
                `(n_samples, n_x, ...)` shaped array where `n_samples` is the
                number of samples and `n_x` is the spatial discretization.
            reducer_labels (1D array, optional): label for X used during the
                fit_transform method for the `dimension_reducer`.
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Returns:
           Reduction of the 2-point statistics of X used to fit the components.

        Example

        >>> from pymks.datasets import make_delta_microstructures
        >>> from pymks import PrimitiveBasis
        >>> n_states = 2
        >>> analyzer = MKSStructureAnalysis(basis=PrimitiveBasis(n_states),
        ...                              n_components=1)
        >>> np.random.seed(5)
        >>> size = (2, 3, 3)
        >>> X = np.random.randint(2, size=size)
        >>>
        >>> print(analyzer.fit_transform(X)) # doctest: +ELLIPSIS
        [[ 0.26731852]
         [-0.26731852]]
        """
        X_stats = self._compute_stats(X, periodic_axes, confidence_index)
        return self._fit_transform(X_stats, None)

    def transform(self, X, periodic_axes=None, confidence_index=None):
        """Computes the 2-point statistics for X and applies dimensionality
        reduction.

        Args:
            X (ND array): The microstructures or spatial correlations, a
                `(n_samples, n_x, ...)` shaped array where `n_samples` is the
                number of samples and `n_x` is the spatial discretization.
            periodic_axes (list, optional): axes that are periodic. (0, 2)
                would indicate that axes x and z are periodic in a 3D
                microstrucure.
            confidence_index (ND array, optional): array with same shape as X
                used to assign a confidence value for each data point.

        Returns:
           Reduction of the 2-point statistics of X.

        Example

        >>> from pymks.datasets import make_delta_microstructures
        >>> from pymks import PrimitiveBasis
        >>> n_states = 2
        >>> analyzer = MKSStructureAnalysis(basis=PrimitiveBasis(n_states),
        ...                              n_components=1)
        >>> np.random.seed(5)
        >>> size = (2, 3, 3)
        >>> X = np.random.randint(2, size=size)
        >>> print(analyzer.fit_transform(X)) # doctest: +ELLIPSIS
        [[ 0.26731852]
         [-0.26731852]]
        >>> print(analyzer.transform(X)) # doctest: +ELLIPSIS
        [[ 0.26731852]
         [-0.26731852]]
        """
        X_stats = self._compute_stats(X, periodic_axes, confidence_index)
        return self._transform(X_stats)

    def _transform(self, X):
        """Reshapes and reduces X"""
        self._store_correlations(X)
        X_reshaped = self._reduce_shape(X)
        self.transform_data =  self.dimension_reducer.transform(X_reshaped)
        return self.transform_data

    def _fit_transform(self, X, y):
        """Reshapes X and uses it to compute the components"""
        if self.store_correlations:
            self.fit_correlations = X
        X_reshaped = self._reduce_shape(X)
        self.fit_data = self.dimension_reducer.fit_transform(X_reshaped, y)
        return self.fit_data

    def _store_correlations(self, X):
        """store stats"""
        if self.store_correlations:
              if hasattr(self, 'transform_correlations'):
                  self.transform_correlations = np.concatenate(
                    (self.transform_correlations, X))
              else:
                  self.transform_correlations = X

    def _compute_stats(self, X, periodic_axes, confidence_index):
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
        """
        if self.basis is None:
            raise AttributeError('basis must be specified')
        X_ = self.basis.discretize(X)
        X_stats = correlate(X_, periodic_axes=periodic_axes,
                            confidence_index=confidence_index,
                            correlations=self.correlations)
        return X_stats

    def _reduce_shape(self, X_stats):
        """
        Helper function used to reshape 2-point statistics appropriately for
        fit and predict methods.

        Args:
            `X_stats`: The discretized microstructure function, an
                `(n_samples, n_x, ..., n_states)` shaped array
                Where `n_samples` is the number of samples, `n_x` is the
                spatial discretization, and n_states is the number of local
                states.

        Returns:
            Spatial correlations for each sample formated with dimensions
            (n_samples, n_features).

        """
        X_reshaped = X_stats.reshape((X_stats.shape[0], X_stats[0].size))
        if self.mean_center:
            X_reshaped -= np.mean(X_reshaped, axis=1)[:, None]
        return X_reshaped
