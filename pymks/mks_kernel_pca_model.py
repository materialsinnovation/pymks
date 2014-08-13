import numpy as np
from sklearn.decomposition import KernelPCA
from pymks.stats import autocorrelate, crosscorrelate
from pymks.bases import DiscreteIndicatorBasis


class MKSKernelPCAModel(KernelPCA):
    """
    The `MKSKernelPCAModel` uses Kernel PCA to quantify the differences between
    microstructures using 2-point statistics. Below is a simple example of
    using the `MKSKernelPCAModel` to find the principle components for a given
    dataset. Let's let the dataset be `X`.

    >>> X = np.array([[0, 0], [1, 1]])

    A basis must be selected for the MKSKernelPCAModel. In this case because
    the dataset is interger values, the `DiscreteIndicatorBasis` can be used.

    >>> basis = DiscreteIndicatorBasis(n_states=2)

    An instance of the `MKSKernelPCAModel` must be created.

    >>> model = MKSKernelPCAModel(basis=basis, n_components=2)

    The principle components for a dataset can be found using the `fit` method.
    Data can be mapped to those principle components using the `transform`
    method or the both the `fit` and `transform` methods can be used on the
    dataset using the `fit_transform` method.

    >>> X_results = model.fit_transform(X)
    >>> X_test = np.array([[-1, 0], [1, 0]])
    >>> assert(np.allclose(X_results, X_test))
    """

    def __init__(self, basis, **kwargs):
        """
        Instantiates a MKSKernalPCAModel.

        Args:
          basis: an instance of a bases class.
          n_components: number of principle n_components. If None, all non-zero
            components are kept.
          kernel: kernel used in kernel PCA
          gamma: Kernel coefficient for rbf and poly kernels. Default:
            1/n_features. Ignored by other kernels.
          degree: Degree for poly kernels. Ignored by other kernels.
          coef0: Independent term in poly and sigmoid kernels. Ignored by other
            kernels.
          kernel_params: Parameters (keyword arguments) and values for kernel
            passed as callable object. Ignored by other kernels.
          alpha: Hyperparameter of the ridge regression that learns the inverse
            transform (when fit_inverse_transform=True).
          fit_inverse_transform: Learn the inverse transform for
            non-precomputed kernels. (i.e. learn to find the pre-image of a
            point)
          eigen_solver: Select eigensolver to use. If n_components is much less
            than the number of training samples, arpack may be more efficient
            than the dense eigensolver.
          tol: convergence tolerance for arpack.
          max_iter: maximum number of iterations for arpack
          remove_zero_eig: If True, then all components with zero eigenvalues
            are removed, so that the number of components in the output may be
            < n_components (and sometimes even zero due to numerical
            instability). When n_components is None, this parameter is ignored
            and components with zero eigenvalues are removed regardless.
        """
        self.basis = basis
        super(MKSKernelPCAModel, self).__init__(**kwargs)

    def _get_spatial_correlations(self, X):
        """
        Generates spatial correlations for a microstructure X.

        Args:
          X: The microstructure, an `(n_samples, N, ...)` shaped
            array where `n_samples` is the number of samples and `N`
            is the spatial discretization.

        Returns:
          Autocorrelations and crosscorrelations.
        """
        X_ = self.basis.discretize(X)
        X_auto = autocorrelate(X_)
        X_cross = crosscorrelate(X_)
        return np.concatenate((X_auto, X_cross), axis=-1)

    def fit(self, X):
        """
        Creates the principle components from data X.

        Args:
          X: The microstructure, an `(n_samples, N, ...)` shaped
            array where `n_samples` is the number of samples and `N`
            is the spatial discretization.
        """
        X_corr = self._get_spatial_correlations(X)
        super(MKSKernelPCAModel, self).fit(self._set_shape(X_corr))

    def transform(self, X):
        """
        Maps data X into principle components.
        Args:
          X: The microstructure, an `(n_samples, N, ...)` shaped
            array where `n_samples` is the number of samples and `N`
            is the spatial discretization.
        Returns:
          Data points for in principle components shaped (n_samples,
          n_components)
        """
        X_corr = self._get_spatial_correlations(X)
        return super(MKSKernelPCAModel,
                     self).transform(self._set_shape(X_corr))

    def fit_transform(self, X):
        """
        Creates the principle components from data X, and maps data X into
        principle components.

        >>> X = np.array([[0, 0], [1, 1]])
        >>> basis = DiscreteIndicatorBasis(n_states=2)
        >>> model = MKSKernelPCAModel(basis=basis, n_components=2)
        >>> X_result = model.fit_transform(X)
        >>> X_test = np.array([[-1., -0.],
        ...                    [ 1., -0.]])
        >>> assert(np.allclose(X_test, X_result))

        Args:
          X: The microstructre function, an `(n_samples, N, ...)` shaped
            array where `n_samples` is the number of samples and `N`
            is the spatial discretization.
        Returns:
          Data points for in principle components shaped (n_samples,
          n_components)
        """
        X_shaped = self._set_shape(X)
        return super(MKSKernelPCAModel, self).fit_transform(X_shaped)

    def _set_shape(self, X_):
        """
        Reshapes data X to (n_samples, n_features).

        >>> X = np.array([[[0, 1],
        ...                [0, 1]],
        ...               [[2, 3],
        ...                [2, 3]]])
        >>> X_test = np.array([[0, 0, 1, 1], [2, 2, 3, 3]])
        >>> basis = DiscreteIndicatorBasis(n_states=4)
        >>> model = MKSKernelPCAModel(basis=basis)
        >>> X_shaped = model._set_shape(X)
        >>> assert(np.allclose(X_shaped, X_test))

        Args:
          X: The microstructure, an `(n_samples, N, ...)` shaped
            array where `n_samples` is the number of samples and `N`
            is the spatial discretization.
        Returns:
          Data shaped (n_samples, n_components)
        """
        size = np.array(X_.shape)
        new_size = (size[0], np.prod(size[1:]))
        return X_.swapaxes(1, -1).reshape(new_size)
