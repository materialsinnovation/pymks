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
    X_auto = autocorrelate(X, basis)

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
    X_auto = autocorrelate(X, basis, periodic_axes=(0, 1))

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
    X_cross = crosscorrelate(X, basis)

    X_result = np.array([[[1 / 3., 4 / 9., 0.5,  4 / 9., ],
                          [1 / 8., 0.25, 3 / 16., 0.25],
                          [0., 2 / 15.,  0., 2 / 15.],
                          [0., 1 / 12., 0, 1 / 12.],
                          [0,   0,    0,   0]]])
    assert(np.allclose(X_result, X_cross[..., 0]))


def test_periodic_crosscorrelation():
    '''
    test periodic crosscorrelation
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import crosscorrelate
    X = np.array([[[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]])
    basis = DiscreteIndicatorBasis(n_states=2)
    X_cross = crosscorrelate(X, basis, periodic_axes=(0, 1))

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
    X_corr = correlate(X, basis)
    X_result = [[2 / 3.,  4 / 9.,  0.75,  4 / 9.],
                [5 / 8.,   0.5,  0.75,   0.5],
                [0.6,  7 / 15.,   0.8,  7 / 15.],
                [5 / 8.,   0.5,  0.75,   0.5],
                [0.5,  4 / 9.,  0.75,  4 / 9.]]
    assert(np.allclose(X_result, X_corr[0, ..., 0]))


def test_periodic_correlate():
    '''
    test corrleate for periodic microstructures
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
    X_corr = correlate(X, basis, periodic_axes=(0, 1))
    X_result = [[0.6,  0.6,  0.75,  0.6],
                [0.6,  0.6,  0.75,  0.6],
                [0.6,  0.6,   0.8,  0.6],
                [0.6,  0.6,  0.75,  0.6],
                [0.6,  0.6,  0.75,  0.6]]
    assert(np.allclose(X_result, X_corr[0, ..., 0]))


def test_periodic_mask():
    '''
    test uncertainty masks for periodic axes.
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import autocorrelate
    from pymks.datasets import make_checkerboard_microstructure

    X = make_checkerboard_microstructure(1, 3)
    basis = DiscreteIndicatorBasis(n_states=2)
    mask = np.ones((X.shape))
    mask[0, 0, 0] = 0
    X_auto_periodic_mask = autocorrelate(X, basis, periodic_axes=[0, 1],
                                         confidence_index=mask)
    X_result_0 = np.array([[[1 / 7., 1 / 7., 3 / 7.],
                          [1 / 7., 0.5, 1 / 7.],
                          [3 / 7., 1 / 7., 1 / 7.]]])
    X_result_1 = np.array([[[2 / 7., 1 / 7., 2 / 7.],
                          [1 / 7., 0.5, 1 / 7.],
                          [2 / 7., 1 / 7., 2 / 7.]]])
    X_result = np.concatenate((X_result_0[..., None],
                               X_result_1[..., None]), axis=-1)
    assert np.allclose(X_auto_periodic_mask, X_result)


def test_nonperiodic_mask():
    '''
    test uncertainty masks for nonperiodic axes.
    '''
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import autocorrelate
    from pymks.datasets import make_checkerboard_microstructure

    X = make_checkerboard_microstructure(1, 3)
    basis = DiscreteIndicatorBasis(n_states=2)
    mask = np.ones((X.shape))
    mask[0, 0, 0] = 0
    X_auto_nonperiodic_mask = autocorrelate(X, basis, confidence_index=mask)
    X_result_0 = np.array([[[1 / 3., 0, 0.5],
                          [0, 0.5, 0.],
                          [0.5, 0, 1 / 3.]]])
    X_result_1 = np.array([[[2 / 3., 0, 0.5],
                          [0, 0.5, 0.],
                          [0.5, 0, 2 / 3.]]])
    X_result = np.concatenate((X_result_0[..., None],
                               X_result_1[..., None]), axis=-1)
    assert np.allclose(X_auto_nonperiodic_mask, np.concatenate(X_result))


def test_mixperdic_mask():
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import autocorrelate
    from pymks.datasets import make_checkerboard_microstructure

    X = make_checkerboard_microstructure(1, 3)
    basis = DiscreteIndicatorBasis(n_states=2)
    mask = np.ones((X.shape))
    mask[0, 0, 0] = 0
    X_auto_mixperiodic_mask = autocorrelate(X, basis, periodic_axes=[0],
                                            confidence_index=mask)
    X_result_0 = np.array([[[1 / 5., 1 / 7., 2 / 5.],
                          [0, 0.5, 0],
                          [2 / 5., 1 / 7., 1 / 5.]]])
    X_result_1 = np.array([[[2 / 5., 1 / 7., 2 / 5.],
                          [0, 0.5, 0.],
                          [2 / 5., 1 / 7., 2 / 5.]]])
    X_result = np.concatenate((X_result_0[..., None],
                               X_result_1[..., None]), axis=-1)
    assert np.allclose(X_auto_mixperiodic_mask, np.concatenate(X_result))


def test_mask_two_samples():
    from pymks import DiscreteIndicatorBasis
    from pymks.stats import correlate
    from pymks.datasets import make_microstructure

    X = make_microstructure(n_samples=2, n_phases=2, size=(3, 3),
                            grain_size=(2, 2), seed=99)
    basis = DiscreteIndicatorBasis(n_states=2)
    mask = np.ones(X.shape)
    mask[:, 0, 0] = 0.
    X_corr = correlate(X, basis, confidence_index=mask)
    X_result = np.array([[[[1 / 3., 1 / 3., 1 / 3.],
                           [1 / 5., 1 / 5., 1 / 5.],
                           [1 / 4., 1 / 4., 0]],
                          [[1 / 5., 1 / 5., 2 / 5.],
                           [1 / 2., 1 / 2., 0],
                           [1 / 5., 1 / 5., 1 / 5.]],
                          [[1 / 4., 1 / 4., 1 / 2.],
                           [1 / 5., 1 / 5., 2 / 5.],
                           [1 / 3., 1 / 3., 0]]],
                         [[[0., 0., 1 / 3.],
                           [2 / 5., 3 / 5., 0.],
                           [0., 0., 1 / 2.]],
                          [[0., 0., 2 / 5.],
                           [3 / 8., 5 / 8., 0],
                           [0., 0., 3 / 5.]],
                          [[0., 0., 1 / 2.],
                           [2 / 5., 3 / 5., 0.],
                           [0., 0., 2 / 3.]]]])
    assert np.allclose(X_corr, X_result)


def test_normalization_rfftn():
    """Test normalization with rfftn
    """
    from pymks import PrimitiveBasis
    from pymks.stats import _normalize
    prim_basis = PrimitiveBasis()
    Nx = Ny = 5
    X_ = np.zeros((1, Nx, Ny, 1))
    prim_basis._axes = np.arange(X_.ndim - 2) + 1
    prim_basis._axes_shape = (2 * Nx, 2 * Ny)
    norm = _normalize(X_.shape, prim_basis, None)
    assert norm.shape == (1, Nx, Ny, 1)
    assert np.allclose(norm[0, Nx / 2, Ny / 2, 0], 25)


def test_normalization_fftn():
    """Test normalization with fftn
    """
    from pymks.bases import FourierBasis
    from pymks.stats import _normalize
    f_basis = FourierBasis()
    Nx = Ny = 5
    X_ = np.zeros((1, Nx, Ny, 1))
    f_basis._axes = np.arange(X_.ndim - 2) + 1
    f_basis._axes_shape = (2 * Nx, 2 * Ny)
    norm = _normalize(X_.shape, f_basis, None)
    assert norm.shape == (1, Nx, Ny, 1)
    assert np.allclose(norm[0, Nx / 2, Ny / 2, 0], 25)


def test_gsh_basis_normalization():
    from pymks.bases import GSHBasis
    from pymks.stats import _normalize
    gsh_basis = GSHBasis()
    Nx = Ny = 5
    X_ = np.zeros((1, Nx, Ny, 1))
    gsh_basis._axes = np.arange(X_.ndim - 2) + 1
    gsh_basis._axes_shape = (2 * Nx, 2 * Ny)
    norm = _normalize(X_.shape, gsh_basis, None)
    assert norm.shape == (1, Nx, Ny, 1)
    assert np.allclose(norm[0, Nx / 2, Ny / 2, 0], 25)


def test_stats_in_parallel():
    import time
    from pymks.bases import PrimitiveBasis
    from pymks.stats import correlate
    from pymks.datasets import make_microstructure
    p_basis = PrimitiveBasis(5)
    if p_basis._pyfftw:
        X = make_microstructure(n_samples=5, n_phases=3)
        t = []
        for i in range(1, 4):
            t_start = time.time()
            correlate(X, p_basis, n_jobs=i)
            t.append(time.time() - t_start)
            assert t == sorted(t, reverse=True)
    else:
        pass


def test_autocorrelate_with_specific_correlations():
    from pymks.stats import autocorrelate
    from pymks import PrimitiveBasis
    X = np.array([[[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 2, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]])
    autocorrelations = [(0, 0), (2, 2)]
    p_basis = PrimitiveBasis(n_states=3)
    X_auto = autocorrelate(X, p_basis, autocorrelations=autocorrelations)
    X_result_0 = np.array([[2 / 3., 1 / 3., 5 / 12., 4 / 9.],
                           [5 / 8., 5 / 12., 9 / 16., 1 / 2.],
                           [1 / 2., 7 / 15., 13 / 20., 7 / 15.],
                           [3 / 8., 1 / 2., 9 / 16., 5 / 12.],
                           [1 / 6., 4 / 9., 5 / 12., 1 / 3.]])
    assert np.allclose(X_auto[0, ..., 0], X_result_0)
    X_result_1 = np.array([[0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0.05, 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.]])
    assert np.allclose(X_auto[0, ..., 1], X_result_1)


def test_crosscorrelate_with_specific_correlations():
    from pymks.stats import crosscorrelate
    from pymks import PrimitiveBasis
    X = np.array([[[0, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 2, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]])
    crosscorrelations = [(1, 2)]
    p_basis = PrimitiveBasis(n_states=3)
    X_cross = crosscorrelate(X, p_basis, crosscorrelations=crosscorrelations)
    X_result = np.array([[0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 1 / 12.],
                         [0., 0., 0., 0.]])
    assert np.allclose(X_cross[0, ..., 0], X_result)


if __name__ == '__main__':
    test_normalization_rfftn()
