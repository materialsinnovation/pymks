"""fMKS - functional materials knowledge system.

See https://github.com/wd15/fmks

fMKS is a functional version of [PyMKS](https://pymks.org) currently
under development. The purpose of the project is to prototype a
parallel implementation of MKS using functional programming in Python
primarily using the [Toolz](http://toolz.readthedocs.io) library.

"""

import os
import pytest
from pkg_resources import get_distribution, DistributionNotFound
from sklearn.base import TransformerMixin, BaseEstimator


def test():  # pragma: no cover
    r"""
    Run all the doctests available.
    """

    path = os.path.split(__file__)[0]
    pytest.main(
        args=[path, "--doctest-modules", "--ignore=setup.py ", "-r s", "--cov=fmks"]
    )


def get_version() -> str:
    """Get the version of the code from egg_info.

    Returns:
      the package version number
    """

    try:
        # pylint: disable=no-member
        version = get_distribution(__name__.split(".", maxsplit=1)[0]).version
    except DistributionNotFound:  # pragma: no cover
        version = "unknown, try running `python setup.py egg_info`"

    return version


__version__ = get_version()

__all__ = ["__version__", "test"]


class GenericTransformer(BaseEstimator, TransformerMixin):
    """Make a generic transformer based on a function

    >>> import numpy as np
    >>> data = np.arange(4).reshape(2, 2)
    >>> GenericTransformer(lambda x: x[:, 1:]).fit(data).transform(data).shape
    (2, 1)
    """

    def __init__(self, func):
        """Instantiate a GenericTransformer

        Function should take a multi-dimensional array and return an
        array with the same length in the sample axis (first axis).

        Args:
          func: transformer function

        """
        self.func = func

    def fit(self, *_):
        """Only necessary to make pipelines work"""
        return self

    def transform(self, data):
        """Transform the data

        Args:
          data: the data to be transformed

        Returns:
          the transformed data
        """
        return self.func(data)
