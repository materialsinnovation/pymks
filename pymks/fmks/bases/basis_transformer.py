"""Base class for LegendreTransformer and PrimitiveTransformer
"""

from sklearn.base import TransformerMixin, BaseEstimator


class BasisTransformer(BaseEstimator, TransformerMixin):
    """Basis transformer for Sklearn pipelines

    Attributes:
      discretize: function to discretize the data
    """

    def __init__(self, discretize, **kwargs):
        """Instantiate a PrimitiveTransformer

        Args:
          discretize: function to discretize the data
          kwargs: args for discretize
        """
        self.discretize = discretize
        self.kwargs = kwargs

    def transform(self, data):
        """Perform the discretization of the data

        Args:
            data: the data to discretize

        Returns:
            the discretized data
        """
        return self.discretize(data, **self.kwargs)

    def fit(self, *_):
        """Only necessary to make pipelines work
        """
        return self
