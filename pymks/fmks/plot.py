"""Plotting functions for PyMKS notebooks
"""
import matplotlib.pyplot as plt
import numpy
from .func import curry, fmap


@curry
def _plot_ax(axis, arrs, titles, cmap, showticks):
    if hasattr(axis.get_subplotspec(), "colspan"):
        col_num = axis.get_subplotspec().colspan.start  # pragma: no cover
    else:
        col_num = axis.colNum  # pragma: no cover
    if not showticks:
        axis.set_xticks(())
        axis.set_yticks(())
    axis.set_title(titles[col_num])
    extent_ = lambda dim: (-arrs[col_num].shape[dim] / 2, arrs[col_num].shape[dim] / 2)
    return axis.imshow(
        arrs[col_num],
        interpolation="none",
        extent=extent_(1) + extent_(0),
        vmin=numpy.min(numpy.vstack(arrs)),
        vmax=numpy.max(numpy.vstack(arrs)),
        cmap=cmap,
    )


def _colorbar(fig, axis, image):
    """Generate a colorbar"""
    axis.yaxis.set_offset_position("right")
    fig.colorbar(image, cax=axis)


def plot_microstructures(
    *arrs, titles=(), cmap=None, colorbar=True, showticks=False, figsize_weight=4
):
    """Plot a set of microstructures side-by-side

    Args:
      arrs: any number of 2D arrays to plot
      titles: a sequence of titles with len(*arrs)
      cmap: any matplotlib colormap

    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x_data = np.random.random((2, 10, 10))
    >>> fig = plot_microstructures(
    ...     x_data[0],
    ...     x_data[1],
    ...     titles=['array 0', 'array 1'],
    ...     cmap='twilight'
    ... )
    >>> fig.show()  #doctest: +SKIP

    .. image:: plot_microstructures.png
       :width: 400


    """
    fig, axs = plt.subplots(
        1,
        len(arrs),
        figsize=(figsize_weight * len(arrs), figsize_weight),
        constrained_layout=True,
    )
    if len(arrs) == 1:
        axs = (axs,)
    if isinstance(titles, str):
        titles = (titles,)
    if len(titles) < len(arrs):
        titles = titles + ("",) * (len(arrs) - len(titles))
    plots = list(
        fmap(_plot_ax(arrs=arrs, titles=titles, cmap=cmap, showticks=showticks), axs)
    )
    if colorbar:
        _colorbar(
            fig,
            fig.add_axes([1.0, 0.05, 0.05, 0.9]),
            plots[0],
        )
    plt.close()
    return fig
