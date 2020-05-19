"""Test functions in func.py
"""

import numpy as np
import dask.array as da

from pymks.fmks.func import make_da


def test_make_da():
    """Test bug in make_da.

    make_da was not preserving the chunks of dask arrays
    """
    arr = np.arange(24).reshape((4, 6))
    darr = da.from_array(arr, chunks=(2, 6))

    def my_func(darr):
        return darr

    new_func = make_da(my_func)

    assert new_func(darr).chunks == ((2, 2), (6,))
