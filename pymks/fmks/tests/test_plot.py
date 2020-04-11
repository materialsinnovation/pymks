"""Test plotting functionality
"""

import matplotlib
import numpy as np
from pymks.fmks.plot import plot_microstructures

matplotlib.use("Agg")


def test_plot():
    """Ensure that the plot_micostructures function is run during the
    tests.

    """
    plot_microstructures(np.arange(4).reshape(2, 2), titles=["test"])
