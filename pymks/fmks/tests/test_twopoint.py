"""
The correlation module test cases

"""
import numpy as np
import dask.array as da

from pymks.fmks.correlations import two_point_stats
from pymks.fmks.correlations import correlations_multiple


# pylint: disable=too-many-arguments
def run_one(size, size_predict, chunk, chunks_predict, periodic_boundary, cutoff):
    """Generic function to test two points stats shape and chunks

    """
    x_data = np.random.randint(2, size=size)
    chunks = (chunk,) + x_data.shape[1:]  # pylint: disable=unsubscriptable-object
    x_data = da.from_array(x_data, chunks=chunks)
    stats = two_point_stats(
        x_data, x_data, periodic_boundary=periodic_boundary, cutoff=cutoff
    )
    assert stats.compute().shape == size_predict
    assert stats.chunks == chunks_predict


def test_twodim_odd():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has odd value height and width
    """
    run_one((2, 15, 15), (2, 9, 9), 1, ((1, 1), (9,), (9,)), False, 4)
    run_one((2, 15, 15), (2, 5, 5), 1, ((1, 1), (5,), (5,)), True, 2)


def test_twodim_even():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has even value height and width
    """
    run_one((3, 10, 10), (3, 9, 9), 2, ((2, 1), (9,), (9,)), False, 4)
    run_one((3, 10, 10), (3, 5, 5), 2, ((2, 1), (5,), (5,)), True, 2)


def test_twodim_mix1():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has odd value height and even value width
    """
    run_one((1, 15, 10), (1, 9, 9), 1, ((1,), (9,), (9,)), False, 4)
    run_one((1, 15, 10), (1, 5, 5), 1, ((1,), (5,), (5,)), True, 2)


def test_twodim_mix2():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has even value height and odd value width
    """
    run_one((1, 10, 15), (1, 9, 9), 1, ((1,), (9,), (9,)), False, 4)
    run_one((1, 10, 15), (1, 5, 5), 1, ((1,), (5,), (5,)), True, 2)


def test_threedim():
    """This test investigates the stability of the two_point_stats funtion with
    3D microstrucure"""
    run_one((4, 10, 10, 15), (4, 9, 9, 9), 2, ((2, 2), (9,), (9,), (9,)), False, 4)
    run_one((4, 10, 10, 15), (4, 5, 5, 5), 2, ((2, 2), (5,), (5,), (5,)), True, 2)


def test_onedim():
    """Test 1D microstructure
    """
    run_one((4, 10), (4, 9), 2, ((2, 2), (9,)), False, 4)
    run_one((4, 10), (4, 5), 2, ((2, 2), (5,)), True, 2)


def test_correlations_multiple():
    """Test that correlations_multiple chunks correctly.

    Test for bug fix. Previously, chunks were ((1, 1), (3,), (3,), (1,
    1)) for this example.

    """
    darr = da.random.randint(2, size=(2, 4, 4, 2), chunks=(1, 4, 4, 2))
    out = correlations_multiple(darr, [[0, 0], [0, 1]])
    assert out.chunks == ((1, 1), (3,), (3,), (2,))
