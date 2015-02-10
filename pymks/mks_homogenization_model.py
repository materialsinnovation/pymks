from pymks.stats import correlate


class MKSHomogenizationModel(object):

    '''
    The `MKSHomogenizationModel` takes in microstructures and a their
    associated macroscopic property, and created a low dimensional structure
    property linkage.

    The `MKSHomogenizationModel` model is designed to integrate with
    dimensionality reduction and techniques
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
