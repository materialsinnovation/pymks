import numpy as np


def test_nonperiodic_autocorrelation():
    '''
    test nonperiodic autocorrelation for spatial statistics
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import autocorrelate
    X = np.array([[[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_ = basis.discretize(X)
    X_auto = autocorrelate(X_)

    X_result = np.array([[[0,       0,       0,       0],
                          [1. / 8, 1. / 12, 3. / 16, 1. / 12],
                          [0.2, 2. / 15,     0.3, 2. / 15],
                          [1. / 8, 1. / 12, 3. / 16, 1. / 12],
                          [0,       0,       0,       0]]])

    assert(np.allclose(X_result, X_auto[..., 1]))


def test_periodic_autocorrelation():
    '''
    test periodic autocorrelation for spatial statistics
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import autocorrelate
    X = np.array([[[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_ = basis.discretize(X)
    X_auto = autocorrelate(X_, periodic_axes=(0, 1))

    X_result = np.array([[[0,   0,    0,   0],
                          [0.1, 0.1, 0.15, 0.1],
                          [0.2, 0.2,  0.3, 0.2],
                          [0.1, 0.1, 0.15, 0.1],
                          [0,   0,    0,   0]]])

    assert(np.allclose(X_result, X_auto[..., 1]))


def test_nonperiodic_crosscorrelation():
    '''
    test nonperiodic crosscorrelation
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import crosscorrelate
    X = np.array([[[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_ = basis.discretize(X)
    X_cross = crosscorrelate(X_)

    X_result = np.array([[[1 / 3., 4 / 9., 0.5,  4 / 9., ],
                          [1 / 8., 0.25, 3 / 16., 0.25],
                          [0., 2 / 15.,  0., 2 / 15.],
                          [0., 1 / 12., 0, 1 / 12.],
                          [0,   0,    0,   0]]])
    assert(np.allclose(X_result, X_cross[..., 0]))


def test_periodic_crosscorrelation():
    '''
    test nonperiodic crosscorrelation
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import crosscorrelate
    X = np.array([[[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_ = basis.discretize(X)
    X_cross = crosscorrelate(X_, periodic_axes=(0, 1))

    X_result = np.array([[[0.3, 0.3, 0.3,  0.3],
                          [0.2, 0.2, 0.15, 0.2],
                          [0.1, 0.1,  0., 0.1],
                          [0.2, 0.2, 0.15, 0.2],
                          [0.3, 0.3, 0.3,  0.3]]])

    assert(np.allclose(X_result, X_cross[..., 0]))


def test_nonperiodic_correlate():
    '''
    test corrleate for non-periodic microstructures
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import correlate

    X = np.array([[[0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0]],
                  [[0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 1, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_ = basis.discretize(X)
    X_corr = correlate(X_)
    X_result = [[0.67,  0.44,  0.75,  0.44],
                [0.62,   0.5,  0.75,   0.5],
                [0.6,  0.47,   0.8,  0.47],
                [0.62,   0.5,  0.75,   0.5],
                [0.5,  0.44,  0.75,  0.44]]
    assert(np.allclose(X_result, X_corr[0, ..., 0]))


def test_nonperiodic_correlate():
    '''
    test corrleate for non-periodic microstructures
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import correlate

    X = np.array([[[0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0]],
                  [[0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 1, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_ = basis.discretize(X)
    X_corr = correlate(X_)
    X_result = [[2 / 3.,  4 / 9.,  0.75,  4 / 9.],
                [5 / 8.,   0.5,  0.75,   0.5],
                [0.6,  7 / 15.,   0.8,  7 / 15.],
                [5 / 8.,   0.5,  0.75,   0.5],
                [0.5,  4 / 9.,  0.75,  4 / 9.]]
    assert(np.allclose(X_result, X_corr[0, ..., 0]))


def test_periodic_correlate():
    '''
    test corrleate for non-periodic microstructures
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import correlate

    X = np.array([[[0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0]],
                  [[0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 1, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_ = basis.discretize(X)
    X_corr = correlate(X_, periodic_axes=(0, 1))
    X_result = [[0.6,  0.6,  0.75,  0.6],
                [0.6,  0.6,  0.75,  0.6],
                [0.6,  0.6,   0.8,  0.6],
                [0.6,  0.6,  0.75,  0.6],
                [0.6,  0.6,  0.75,  0.6]]
    assert(np.allclose(X_result, X_corr[0, ..., 0]))


if __name__ == '__main__':
    test_periodic_crosscorrelation()
    test_nonperiodic_crosscorrelation()
    test_periodic_autocorrelation()
    test_nonperiodic_autocorrelation()
