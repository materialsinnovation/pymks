import numpy as np
from pymks.stats import correlate
from sklearn.base import BaseEstimator
from sklearn.decomposition import RandomizedPCA


class MKSStructureAnalysis(BaseEstimator):
    """
    """
    def __init__(self, basis, correlations=None, dimension_reducer=None,
                 n_components=None, store_correlations=False):
        self.basis = basis
        self.correlations = correlations
        self.dimension_reducer = dimension_reducer
        self.store_correlations = store_correlations
        if self.dimension_reducer is None:
            self.dimension_reducer = RandomizedPCA(copy=False)
        if n_components is None:
            n_components = self.dimension_reducer.n_components
        if n_components is None:
            n_components = 5
        self.n_components = n_components
        if self.correlations is None and basis is not None:
            self.correlations = [(0, l) for l in range(basis.n_states)]
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

    def fit(self, X, periodic_axes=None, confidence_index=None):
        X_stats = self._compute_stats(self, X, periodic_axes, confidence_index)
        if self.store_correlations:
            self.fit_stats = X_stats
        X_stats = self._reduce_shape(X_stats.reshape[0], -1)
        self.fit_data = self._fit(X_stats)

    def transform(self, X, periodic_axes, confidence_index):
        X_stats = self._compute_stats(self, X, periodic_axes, confidence_index)
        if self.store_correlations:
            self.transform_stats = X_stats
        self.transforme_data = self._transform(X_stats)

    def fit_transform(self, X, periodic_axes=None, confidence_index=None):
        X_stats = self._compute_stats(self, X, periodic_axes, confidence_index)
        if self.store_correlations:
            self.fit_stats = X_stats
        return self._fit_transform(X_stats, None)

    def _fit_transform(self, X, y):
        X_reshaped = self._reduce_shape(X)
        self.fit_data = self.dimension_reducer.fit_transform(X_reshaped, y)
        return self.fit_data

    def _transform(self, X):
        X_reshaped = self._reduce_shape(X)
        return self.dimension_reducer.transform(X_reshaped)

    def _fit(self, X, y):
        X_reshaped = self._reduce_shape(X)
        self.dimension_reducer.fit(X_reshaped, y)

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
                Where `n_samples` is the number of samples, `n_x` is thes
                patial discretization, and n_states is the number of local
                states.

        Returns:
            Spatial correlations for each sample formated with dimensions
            (n_samples, n_features).

        """
        X_reshaped = X_stats.reshape((X_stats.shape[0], X_stats[0].size))
        return X_reshaped - np.mean(X_reshaped, axis=1)[:, None]
