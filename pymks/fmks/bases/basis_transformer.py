"""Base class for LegendreTransformer and PrimitiveTransformer
"""

from sklearn.base import TransformerMixin, BaseEstimator
import dask.array as da


class BasisTransformer(BaseEstimator, TransformerMixin):
    """Basis transformer for Sklearn pipelines

    Attributes:
      discretize: function to discretize the data
      n_state: the number of local states
      min_: the minimum local state
      max_: the maximum local state
      chunks: chunks size for state axis

    >>> import numpy as np
    >>> f = lambda *_, **__: None
    >>> BasisTransformer(f).fit().transform(np.arange(4).reshape(1, 2, 2))

    """

    # pylint: disable=too-many-arguments

    def __init__(self, discretize, n_state=2, min_=0.0, max_=1.0, chunks=None):
        """Instantiate a PrimitiveTransformer
        """
        self.discretize = discretize
        self.n_state = n_state
        self.min_ = min_
        self.max_ = max_
        self.chunks = chunks

    def transform(self, data):
        """Perform the discretization of the data

        Args:
            data: the data to discretize

        Returns:
            the discretized data
        """
        return self.discretize(
            data if hasattr(data, "chunks") else da.from_array(data, data.shape),
            n_state=self.n_state,
            min_=self.min_,
            max_=self.max_,
            chunks=self.chunks,
        )

    def fit(self, *_):
        """Only necessary to make pipelines work
        """
        return self
