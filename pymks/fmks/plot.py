"""Plotting functions for PyMKS notebooks
"""
import matplotlib.pyplot as plt
import numpy
from .func import curry, fmap


@curry
def _plot_ax(axis, arrs, titles, cmap):
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_title(titles[axis.colNum])
    return axis.imshow(
        arrs[axis.colNum],
        interpolation="none",
        vmin=numpy.min(numpy.vstack(arrs)),
        vmax=numpy.max(numpy.vstack(arrs)),
        cmap=cmap,
    )


def _colorbar(fig, axis, image):
    """Generate a colorbar
    """
    axis.yaxis.set_offset_position("right")
    fig.colorbar(image, cax=axis)


def plot_microstructures(*arrs, titles=(), cmap=None, colorbar=True):
    """Plot a set of microstructures

    Args:
      *arrs: any number of 2D arrays to plot
      titles: a sequence of titles with len(*arrs)
      cmap: any matplotlib colormap

    """
    fig, axs = plt.subplots(
        1, len(arrs), figsize=(4 * len(arrs), 4), constrained_layout=True
    )
    if len(arrs) == 1:
        axs = (axs,)
    if isinstance(titles, str):
        titles = (titles,)
    if len(titles) < len(arrs):
        titles = titles + ("",) * (len(arrs) - len(titles))
    plots = list(fmap(_plot_ax(arrs=arrs, titles=titles, cmap=cmap), axs))
    if colorbar:
        _colorbar(
            fig, fig.add_axes([1.0, 0.05, 0.05, 0.9]), plots[0],
        )
