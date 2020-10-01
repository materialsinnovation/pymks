"""MKS Pair Correlation Module
For computing pair correlations from
two point statistics.
"""
import numpy as np
import dask.array as da


def dist_from_center(center_arr, axes=None):
    """
    Calculate the distance from the center pixel of a
    centered array (like 2point statistics).
    Supports any number of dimensions.

    Args:
      center_arr: Centered array. Must support ndim and
        shape methods (like numpy arrays).
      axes: axes which should be included in the distance calculation.

    Returns:
      Distances array of same shape as center_arr.

    >>> from pymks.fmks.correlations import auto_correlation
    >>> center_arr = da.random.randint(0,2, [2,3,4])
    >>> distance = dist_from_center(center_arr, axes=[1,2])
    >>> tps = auto_correlation(center_arr).compute()
    >>> assert len(np.argwhere(distance==0)) == distance.shape[0]
    >>> assert all(tps[distance==0] == np.amax(tps, axis=(1,2)))
    """

    if axes is None:
        axes = set(range(center_arr.ndim))

    # New shape has a size of 1 in the ignored axes
    new_shape = [s if i in axes else 1 for i, s in enumerate(center_arr.shape)]

    # Build matrices of x, y, z, etc coordinates from center with meshgrid
    args = [
        np.linspace(
            -(center_arr.shape[i] // 2),
            center_arr.shape[i] // 2 + center_arr.shape[i] % 2 - 1,
            center_arr.shape[i],
        )
        for i in range(0, center_arr.ndim)
        if i in axes
    ]

    # Sum over the squared coordinates
    distance = sum([a ** 2 for a in np.meshgrid(*args, indexing="ij")])

    # Take square root for euclidean distance and reshape to appropriate
    # number of dimensions. Then broadcast to the shape of input.
    return np.broadcast_to(np.sqrt(distance).reshape(new_shape), center_arr.shape)


def paircorr_from_twopoint(twopoint, cutoff_r=None, interpolate_n=None):
    """
    Computes the pair correlations from 2point statistics. Assumes that
    each pixel is one unit for radius calculation. If interpolating, this
    function uses linear interpolation. If another interpolation is desired,
    don't specify this parameter and perform desired interpolation on output.

    Args:
      twopoint: A centered 2point statistic array. (n_sample, statistic dimensions)
      cutoff_r: Float. Return radii and probabilities beneath this value.
        Values greater than 1 are used as an exact radius cutoff. Values
        less than 1 are treated as a fraction of the maximum radius.
      interpolate_n: Int. Specify the number of equally spaced radii that
        the probabilities will be interpolated to.

    Returns:
      Pair Correlations Array (interpolate_n, n_samples+1). paircorr[:,0]
      is the radius values. paircorr[:,1:] are the probabilities, where
      paircorr[:,1] corresponds to the sample in twopoint[0,...].


    >>> center_arr = np.array([
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
    >>> pc_correct = np.array([
    ...     [0, 0.5, 0.6],
    ...     [1, 0.45, 0.3],
    ...     [np.sqrt(2), 0.25, 0.2]
    ... ])
    >>> pc_correct_cut = np.array([
    ...     [0, 0.5, 0.6],
    ...     [1, 0.45, 0.3]
    ... ])
    >>> pc_correct_interped = np.array([
    ...     [0, 0.5, 0.6],
    ...     [np.sqrt(2), 0.25, 0.2]
    ... ])
    >>> assert np.allclose(paircorr_from_twopoint(center_arr), pc_correct)
    >>> assert np.allclose(paircorr_from_twopoint(center_arr,
    ...     cutoff_r=1.01), pc_correct_cut)
    >>> assert np.allclose(paircorr_from_twopoint(center_arr, cutoff_r=0.99),
    ...     pc_correct_cut)
    >>> assert np.allclose(paircorr_from_twopoint(center_arr, interpolate_n=2),
    ...     pc_correct_interped)
    """

    distance = dist_from_center(twopoint[0, ...])

    distance = distance.reshape(-1)
    twopoint = twopoint.reshape(twopoint.shape[0], -1)

    ind_sort = np.argsort(distance)

    distance = distance[ind_sort]
    twopoint = twopoint[:, ind_sort]

    radii, ind_split = np.unique(distance, return_index=True)

    probs = np.array(
        [np.average(arr, axis=1) for arr in np.split(twopoint, ind_split[1:], axis=1)]
    )

    if cutoff_r is None:
        pass

    elif cutoff_r > 1:
        r_inds = radii <= cutoff_r

        radii = radii[r_inds]
        probs = probs[r_inds, :]

    elif cutoff_r < 1:
        r_inds = radii <= (cutoff_r * radii[-1])

        radii = radii[r_inds]
        probs = probs[r_inds, :]

    if interpolate_n:
        radii_out = np.linspace(radii.min(), radii.max(), interpolate_n)
        probs_out = np.zeros((len(radii_out), probs.shape[1]))

        for i in range(probs.shape[1]):
            probs_out[:, i] = np.interp(radii_out, radii, probs[:, i])

        return da.array(np.concatenate([radii_out.reshape(-1, 1), probs_out], axis=1))

    return da.array(np.concatenate([radii.reshape(-1, 1), probs], axis=1))
