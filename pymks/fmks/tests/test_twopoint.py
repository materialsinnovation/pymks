"""
The correlation module test cases

"""
import numpy as np
import dask.array as da

from pymks.fmks.correlations import two_point_stats


def test_twodim_odd():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has odd value height and width
    """
    np.random.seed(122)
    x_data = np.asanyarray([np.random.randint(2, size=(15, 15))])
    chunks = x_data.shape
    x_data = da.from_array(x_data, chunks=(chunks))
    stats = two_point_stats(x_data, x_data, periodic_boundary=False, cutoff=4)
    stats2 = two_point_stats(x_data, x_data, periodic_boundary=True, cutoff=2)
    assert stats.compute().shape == (1, 9, 9)
    assert stats2.compute().shape == (1, 5, 5)


def test_twodim_even():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has even value height and width
    """
    np.random.seed(122)
    x_data = np.asanyarray([np.random.randint(2, size=(10, 10))])
    chunks = x_data.shape
    x_data = da.from_array(x_data, chunks=(chunks))
    stats = two_point_stats(x_data, x_data, periodic_boundary=False, cutoff=4)
    stats2 = two_point_stats(x_data, x_data, periodic_boundary=True, cutoff=2)
    assert stats.compute().shape == (1, 9, 9)
    assert stats2.compute().shape == (1, 5, 5)


def test_twodim_mix1():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has odd value height and even value width
    """

    np.random.seed(122)
    x_data = np.asanyarray([np.random.randint(2, size=(15, 10))])
    chunks = x_data.shape
    x_data = da.from_array(x_data, chunks=(chunks))
    stats = two_point_stats(x_data, x_data, periodic_boundary=False, cutoff=4)
    stats2 = two_point_stats(x_data, x_data, periodic_boundary=True, cutoff=2)
    assert stats.compute().shape == (1, 9, 9)
    assert stats2.compute().shape == (1, 5, 5)


def test_twodim_mix2():
    """This test investigates the stability of the two_point_stats funtion with
    a microstrucure that has even value height and odd value width
    """
    np.random.seed(122)
    x_data = np.asanyarray([np.random.randint(2, size=(10, 15))])
    chunks = x_data.shape
    x_data = da.from_array(x_data, chunks=(chunks))
    stats = two_point_stats(x_data, x_data, periodic_boundary=False, cutoff=4)
    stats2 = two_point_stats(x_data, x_data, periodic_boundary=True, cutoff=2)
    assert stats.compute().shape == (1, 9, 9)
    assert stats2.compute().shape == (1, 5, 5)


def test_threedim():
    """This test investigates the stability of the two_point_stats funtion with
    3D microstrucure"""
    np.random.seed(122)
    x_data = np.asanyarray([np.random.randint(2, size=(10, 10, 15))])
    chunks = x_data.shape
    x_data = da.from_array(x_data, chunks=(chunks))
    stats = two_point_stats(x_data, x_data, periodic_boundary=False, cutoff=4)
    stats2 = two_point_stats(x_data, x_data, periodic_boundary=True, cutoff=2)
    assert stats.compute().shape == (1, 9, 9, 9)
    assert stats2.compute().shape == (1, 5, 5, 5)
