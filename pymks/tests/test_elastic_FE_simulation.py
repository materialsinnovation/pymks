

def test_issue106():
    """
    Using a doctest to check the error strings.

    >>> from pymks.datasets.elastic_FE_simulation import ElasticFESimulation
    >>> import numpy as np
    >>> L = 5
    >>> elastic_modulus = (1, 2, 3)
    >>> poissons_ratio = (0.3, 0.3, 0.3)
    >>> size = (1, L, L)
    >>> sim = ElasticFESimulation(elastic_modulus=elastic_modulus,
    ...                           poissons_ratio=poissons_ratio)
    >>> X = np.zeros(size, dtype=int)
    >>> sim.run(X)
    >>> X = np.ones(size, dtype=int)
    >>> sim.run(X)
    >>> X[0, 0, 0] = -1
    >>> sim.run(X)
    Traceback (most recent call last):
    ...
    RuntimeError: X must be between 0 and 2.
    """
    pass
