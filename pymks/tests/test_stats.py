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

    X_result = np.array([[[1 / 3., 4 / 9., 0.5,  4 / 9, ],
                          [1 / 8., 0.25, 3 / 16, 0.25],
                          [0., 1 / 7.,  0., 1 / 7.],
                          [0., 1 / 20., 0.15, 1 / 20.],
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


if __name__ == '__main__':
    test_periodic_crosscorrelation()
