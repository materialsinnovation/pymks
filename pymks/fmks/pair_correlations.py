"""MKS Pair Correlation Module
For computing pair correlations from
two point statistics.
"""
import numpy as np
from pymks.fmks.func import (
    fmap,
    sequence,
    star,
    dist_mesh,
    sort_array,
    make_da,
    apply_dict_func,
    curry,
)


def calc_probs(x_data, ind_sort, ind_split):
    """Compute the two point probabilities averages radially

    Args:
       x_data: the two point statistics
       ind_sort: the sorted index of distances from the center
       ind_split: the index groupings

    Returns:
       the calculated probabilites averaged over each radii
       grouping

    """
    return sequence(
        lambda x: x.reshape(x.shape[0], -1)[:, ind_sort],
        lambda x: np.split(x, ind_split[1:], axis=1),
        fmap(lambda x: np.average(x, axis=1)),
        list,
        np.array,
    )(x_data)


def calc_radii(shape):
    """Computer the radii with unique distances from the center

    Args:
      shape: the shape of the domain

    Returns:
      a tuple of (radii, index_split, index_sort) where the
      index_split are groupings of indices at the same distance
      and index_sort are the sorted indices
    """
    flatten = lambda x: x.reshape(-1)
    return sequence(
        dist_mesh,
        flatten,
        sort_array,
        lambda x: np.unique(x[0], return_index=True) + (x[1],),
    )(shape)


def r_inds(cutoff_r, radii):
    """Workout the maximum radii to include

    Args:
      cutoff_r: the cutoff radius (if less than 1 assume it's a
        proportion)
      radii: the radii

    Returns:
      an index of True and False to do mask the proabilities and radii
      for further calculation

    """
    if cutoff_r is None:
        return slice(None)
    if cutoff_r > 1:
        return radii <= cutoff_r
    return radii <= (cutoff_r * radii[-1])


@curry
def calc_cutoff(cutoff_r, probs, radii):
    """Perform the cutoff baseed on cutoff_r

    See `r_inds` above
    """
    index = r_inds(cutoff_r, radii)
    return probs[index], radii[index]


@curry
def interpolate(interpolate_n, probs, radii):
    """Interpoleate the probailites to evenly spaced radii

    Using the max and min radii with `interpolate_n` elements
    interpolate the previously caculated probabilites to new spacing.

    Args:
      interpolate_n: the number of elements
      probs: the probabilites
      radii: the radii

    Returns:
      Tuple of new probabilites and radii

    """
    if interpolate_n is None:
        return probs, radii

    radii_interp = np.linspace(radii.min(), radii.max(), interpolate_n)
    interp = lambda x: np.interp(radii_interp, radii, x)
    return (
        sequence(np.transpose, fmap(interp), list, np.array, np.transpose)(probs),
        radii_interp,
    )


@curry
def np_paircorr(x_data, cutoff_r=None, interpolate_n=None):
    """Numpy only version of `paircorr_from_twopoint`

    """
    return sequence(
        calc_radii,
        lambda x: (calc_probs(x_data, x[2], x[1]), x[0]),
        star(calc_cutoff(cutoff_r)),
        star(interpolate(interpolate_n)),
        lambda x: (np.transpose(x[0]), x[1]),
    )(x_data[0].shape)


@make_da
def paircorr_from_twopoint(x_data, cutoff_r=None, interpolate_n=None):
    r"""Computes the pair correlations from 2-point statistics.

    The pair correlations are the radial average of the 2 point
    stats. The grid spacing is assumed to be one unit. Linear
    interpolation is used if ``interpolate_n`` is specified. If
    another interpolation is desired, don't specify this parameter and
    perform desired interpolation on the output.

    The discretized two point statistics are given by

    $$ f[r \\; \\vert \\; l, l'] = \\frac{1}{S} \\sum_s m[s, l] m[s + r, l'] $$

    where $ f[r \\; \\vert \\; l, l'] $ is the conditional probability of
    finding the local states $l$ and $l'$ at a distance and
    orientation away from each other defined by the vector $r$. `See
    this paper for more details on the
    notation. <https://doi.org/10.1007/s40192-017-0089-0>`_

    The pair correlation is defined as the conditional probability for
    the case of the magnitude vector, $||r||_2$, defined by $ g[ d ]$.
    $g$ is related to $f$ via the following
    transformation. Consider the set, $ I[d] := \\{ f[r] \\; \\vert \\; ||r||_2 = d \\} $
    then

    $$ g[d] = \\frac{1}{ | I[ d ] | }  \\sum_{f \\in I[ d ]} f $$

    The $d$ are radii from the center pixel of the domain. They are
    automatially calculated if ``interpolate_n`` is ``None``.

    It's assumed that ``x_data`` is a valid set of two point statistics
    calculated from the PyMKS correlations module.

    Args:
      x_data: array of centered 2-point statistics. (n_samples, n_x,
        n_y, ...)
      cutoff_r: the radius cut off. Values less than 1 are assumed to
        be a proportion while values greater than 1 are an exact
        radius cutoff
      interpolate_n: the number of equally spaced radii that the
        probabilities will be interpolated to

    Returns:
      A tuple of the pair correlation array and the radii cutoffs used
      for averaging or interpolation. The pair correlations are shaped
      as ``(n_samples, n_radii)``, whilst the radii are shaped as
      ``(n_radii,)``. ``n_radii`` is equal to ``interpolate_n`` when
      ``interpolate_n`` is specified. The probabilities are chunked on
      the sample axis the same as ``x_data``. The radii is a numpy array.

    Test with only 2 samples of 3x3

    >>> x_data = np.array([
    ...     [
    ...         [0.2, 0.4, 0.3],
    ...         [0.4, 0.5, 0.5],
    ...         [0.2, 0.5, 0.3]
    ...     ],
    ...     [
    ...         [0.1, 0.2, 0.3],
    ...         [0.2, 0.6, 0.4],
    ...         [0.1, 0.4, 0.3]
    ...     ]
    ... ])

    Most basic test

    >>> probs, radii = paircorr_from_twopoint(x_data)
    >>> assert np.allclose(probs,
    ...     [[0.5, 0.45, 0.25],
    ...      [0.6, 0.3, 0.2]])
    >>> assert np.allclose(radii, [0, 1, np.sqrt(2)])

    Test with ``cutoff_r`` greater than 1

    >>> probs, radii = paircorr_from_twopoint(x_data, cutoff_r=1.01)
    >>> assert np.allclose(probs,
    ...     [[0.5, 0.45],
    ...      [0.6, 0.3]])
    >>> assert np.allclose(radii, [0, 1])

    Test with ``cutoff_r`` less than 1

    >>> probs, radii = paircorr_from_twopoint(x_data, cutoff_r=0.99)
    >>> assert np.allclose(probs,
    ...     [[0.5, 0.45],
    ...      [0.6, 0.3]])
    >>> assert np.allclose(radii, [0, 1])

    Test with a linear interpolation

    >>> probs, radii = paircorr_from_twopoint(x_data, interpolate_n=2)
    >>> assert np.allclose(probs,
    ...     [[0.5, 0.25],
    ...      [0.6, 0.2]])
    >>> assert np.allclose(radii, [0, np.sqrt(2)])

    Test with Dask. The chunks along the sample axis are preserved.

    >>> arr = da.from_array(np.random.random((10, 4, 3, 3)), chunks=(2, 4, 3, 3))
    >>> probs, radii = paircorr_from_twopoint(arr)
    >>> probs.shape
    (10, 7)
    >>> probs.chunks
    ((2, 2, 2, 2, 2), (7,))
    >>> assert np.allclose(radii, np.sqrt([0, 1, 2, 3, 4, 5, 6]))

    """
    func = np_paircorr(cutoff_r=cutoff_r, interpolate_n=interpolate_n)
    ## Do one sample in serial to calculate the radii and probs shape
    probs, radii = func(x_data[:1].compute())

    return (
        apply_dict_func(
            lambda x: dict(probs=func(x)[0]), x_data, dict(probs=probs.shape)
        )["probs"],
        radii,
    )
