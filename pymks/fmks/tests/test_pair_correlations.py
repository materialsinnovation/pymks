"""Test pair correlations
"""

import dask.array as da
import numpy as np
from pymks.fmks import correlations


def get_array():
    """Get array for tests
    """
    return da.from_array(
        np.array([[[1, 0, 0], [0, 1, 1], [1, 1, 0]], [[0, 0, 1], [1, 0, 0], [0, 0, 1]]])
    )


def get_mask():
    """Get mask for tests
    """
    mask = np.ones((2, 3, 3))
    mask[:, 2, 1:] = 0
    return da.from_array(mask)


def test_periodic():
    """Check that periodic still works

    The normalization occurs in the two_point_stats function and the
    auto-correlation/cross-correlation occur in the cross_correlation
    function. Checking that the normalization is properly calculated.
    First is the auto-correlation. Second is the cross-correlation.
    """
    array = get_array()
    correct = (
        (correlations.cross_correlation(array, array).compute() / 9)
        .round(3)
        .astype(np.float64)
    )
    tested = (
        correlations.two_point_stats(array, array).compute().round(3).astype(np.float64)
    )
    assert (correct == tested).all()

    correct = (
        (correlations.cross_correlation(array, 1 - array).compute() / 9)
        .round(3)
        .astype(np.float64)
    )
    tested = (
        correlations.two_point_stats(array, 1 - array)
        .compute()
        .round(3)
        .astype(np.float64)
    )
    assert (correct == tested).all()


def test_masked_periodic():
    """Check that masked periodic works

    Two point statistics are part correlation and part
    normalization. The correlation sums up the number of possible
    2-point states. In masked periodic, we assume that vectors going
    across the boundary of the structure come back on the other
    side. However, a vector landing in the masked area is discarded
    (ie not included in the correlation sum). Below, are the hand
    computed correlation and normalization. The correct 2point stats
    are the correlation divided by the normalization. First, is the
    auto-correlation and second is the cross-correlation.
    """

    correct_periodic_mask_auto = np.array(
        [[[2, 1, 2], [1, 4, 1], [2, 1, 2]], [[1, 0, 0], [0, 2, 0], [0, 0, 1]]]
    )

    correct_periodic_mask_cross = np.array(
        [[[1, 3, 1], [2, 0, 2], [1, 1, 1]], [[0, 1, 2], [2, 0, 2], [1, 2, 0]]]
    )

    norm_periodic_mask = np.array([[5, 5, 5], [6, 7, 6], [5, 5, 5]])

    mask = get_mask()
    array = get_array()

    # Auto-Correlation
    correct = (
        (correct_periodic_mask_auto / norm_periodic_mask).round(3).astype(np.float64)
    )
    tested = (
        correlations.two_point_stats(array, array, mask=mask, periodic_boundary=True)
        .compute()
        .round(3)
        .astype(np.float64)
    )

    assert (correct == tested).all()

    # Cross-Correlation
    correct = (
        (correct_periodic_mask_cross / norm_periodic_mask).round(3).astype(np.float64)
    )
    tested = (
        correlations.two_point_stats(
            array, 1 - array, mask=mask, periodic_boundary=True
        )
        .compute()
        .round(3)
        .astype(np.float64)
    )

    assert (correct == tested).all()


def test_non_periodic():
    """Test that non-periodic works

    Two point statistics are part correlation and part
    normalization. The correlation sums up the number of possible
    2-point states. In non-periodic, we assume that a vector used to
    count up 2 point states can only connect two states in the
    structure. A vector going outside of the bounds of the structure
    is not counted.

    Below, are the hand computed correlation and normalization. The
    correct 2point stats are the correlation divided by the
    normalization. First, is the auto-correlation and second is the
    cross-correlation.

    """
    correct_nonperiodic_auto = np.array(
        [[[1, 1, 2], [2, 5, 2], [2, 1, 1]], [[0, 0, 0], [0, 3, 0], [0, 0, 0]]]
    )

    correct_nonperiodic_cross = np.array(
        [[[2, 3, 1], [1, 0, 2], [0, 2, 1]], [[1, 2, 1], [2, 0, 1], [1, 2, 1]]]
    )

    norm_nonperiodic = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]])

    array = get_array()
    # Auto-Correlation
    correct = (correct_nonperiodic_auto / norm_nonperiodic).round(3).astype(np.float64)
    tested = (
        correlations.two_point_stats(array, array, periodic_boundary=False)
        .compute()
        .round(3)
        .astype(np.float64)
    )

    assert (correct == tested).all()

    # Cross-Correlation
    correct = (correct_nonperiodic_cross / norm_nonperiodic).round(3).astype(np.float64)
    tested = (
        correlations.two_point_stats(array, 1 - array, periodic_boundary=False)
        .compute()
        .round(3)
        .astype(np.float64)
    )

    assert (correct == tested).all()


def test_non_periodic_masking():
    """Check that non-periodic masking works

    In non-periodic masking, vectors that go across the boundary or
    land in a mask are not included in the sum.

    """

    correct_nonperiodic_mask_auto = np.array(
        [[[1, 0, 1], [1, 4, 1], [1, 0, 1]], [[0, 0, 0], [0, 2, 0], [0, 0, 0]]]
    )

    correct_nonperiodic_mask_cross = np.array(
        [[[1, 3, 1], [1, 0, 1], [0, 1, 0]], [[0, 1, 1], [1, 0, 1], [1, 2, 0]]]
    )

    norm_nonperiodic_mask = np.array([[2, 4, 3], [4, 7, 4], [3, 4, 2]])

    array = get_array()
    mask = get_mask()

    # Auto-Correlation
    correct = (
        (correct_nonperiodic_mask_auto / norm_nonperiodic_mask)
        .round(3)
        .astype(np.float64)
    )
    tested = (
        correlations.two_point_stats(array, array, mask=mask, periodic_boundary=False)
        .compute()
        .round(3)
        .astype(np.float64)
    )
    assert (correct == tested).all()

    # Cross-Correlation
    correct = (
        (correct_nonperiodic_mask_cross / norm_nonperiodic_mask)
        .round(3)
        .astype(np.float64)
    )
    tested = (
        correlations.two_point_stats(
            array, 1 - array, mask=mask, periodic_boundary=False
        )
        .compute()
        .round(3)
        .astype(np.float64)
    )
    assert (correct == tested).all()


def test_different_sized_arrays():
    """Check that different sized dask arrays are valid masks.

    We want to be able to specify the same mask for each sample. We
    also want to be able to specify a different mask for each
    sample. This validates that both are possible.
    """

    array = da.random.random([1000, 3, 3])

    mask_same4all = da.random.randint(0, 2, [3, 3])
    mask_diff4all = da.random.randint(0, 2, [1000, 3, 3])

    correlations.two_point_stats(array, array, mask=mask_same4all)

    # The following check fails. Therefore, the current implementation
    # only works for one mask for all or different mask for all, which
    # is feature rich enough for me.
    # correlations.two_point_stats(array, array, mask=mask_same4some)
    correlations.two_point_stats(array, array, mask=mask_diff4all)


def test_bools_ints_valid():
    """Some check that boolean and integers are valid masks

    A mask could be true and false specifying where there is a
    microstructure. However, it could also be any value in the range
    $[0,1]$ which specifies the probability a value is correctly
    assigned. The mask right now only implements confidence in a
    single phase, although idealy it should represent the confidence
    in all phases. However, for the use cases where there are 2
    phases, a mask with a probability for one phase also completely
    describes the confidence in the other phase. Therefore, this
    implementation is complete for 2 phases.

    """

    mask_int = da.random.randint(0, 2, [1000, 3, 3])
    mask_bool = mask_int.copy().astype(bool)
    array = da.random.random([1000, 3, 3])

    correlations.two_point_stats(array, array, mask=mask_int)
    correlations.two_point_stats(array, array, mask=mask_bool)
