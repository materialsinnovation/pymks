import numpy as np


def dist_from_center(C, axes=None):
    '''
    Calculate the distance from the center pixel of a
    centered array (like 2point statistics).
    Supports any number of dimensions.

    Args:
      C: Centered array. Must support ndim and
        shape methods (like numpy arrays).
      axes: axes which should be included in the distance calculation.

    Returns:
      Distances array of same shape as C.

    >>> C = da.random.randint(0,2, [2,3,4])
    >>> D = dist_from_center(C, axes=[1,2])
    >>> tps = auto_correlation(C).compute()
    >>> assert len(np.argwhere(D==0)) == D.shape[0]
    >>> assert all(tps[D==0] == np.amax(tps, axis=(1,2)))
    '''

    if axes is None:
        axes = set(range(C.ndim))

    # New shape has a size of 1 in the ignored axes
    new_shape = [s if i in axes else 1 for i, s in enumerate(C.shape)]

    # Build matrices of x, y, z, etc coordinates from center with meshgrid
    args = [
        np.linspace(
            -(C.shape[i]//2),
            C.shape[i]//2 + C.shape[i]%2 - 1,
            C.shape[i]
        )
        for i in range(0, C.ndim) if i in axes
    ]

    # Sum over the squared coordinates
    D = sum([a**2 for a in np.meshgrid(*args, indexing='ij')])

    # Take square root for euclidean distance and reshape to appropriate
    # number of dimensions. Then broadcast to the shape of input.
    return np.broadcast_to(np.sqrt(D).reshape(new_shape), C.shape)


def paircorr_from_twopoint_1sample(G, cutoff_r=None, interpolate_n=None):
    '''
    Computes the pair correlations from 2point statistics. Assumes that
    each pixel is one unit for radius calculation. If interpolating, this
    function uses linear interpolation. If another interpolation is desired,
    don't specify this parameter and perform desired interpolation on output.

    Args:
      G: A centered 2point statistic array.
      cutoff_r: Float. Return radii and probabilities beneath this value.
        Values greater than 1 are used as an exact radius cutoff. Values
        less than 1 are treated as a fraction of the maximum radius.
      interpolate_n: Int. Specify the number of equally spaced radii that
        the probabilities will be interpolated to.

    Returns:
      Pair Correlations Array (n, 2). paircorr[:,0] is the radius values.
      paircorr[:,1] is the probabilities.



    '''
    D = dist_from_center(G)

    D = D.ravel()
    G = G.ravel()

    ind_sort = np.argsort(D)

    D = D[ind_sort]
    G = G[ind_sort]

    radii, ind_split = np.unique(D, return_index=True)

    probs = np.array([np.average(arr) for arr in np.split(G, ind_split[1:])])

    if cutoff_r is None:
        pass

    elif cutoff_r>1:
        r_inds = radii <= cutoff_r

        radii = radii[r_inds]
        probs = probs[r_inds]

    elif cutoff_r<1:
        r_inds = radii <= (cutoff_r*radii[-1])

        radii = radii[r_inds]
        probs = probs[r_inds]

    if interpolate_n:
        radii_out = np.linspace(radii.min(), radii.max(), interpolate_n)
        probs_out = np.interp(radii_out, radii, probs)

        return np.stack([radii_out, probs_out], axis=1)
    else:
        return np.stack([radii, probs], axis=1)


def paircorr_from_twopoint(G, cutoff_r=None, interpolate_n=None):
    '''
    Computes the pair correlations from 2point statistics. Assumes that
    each pixel is one unit for radius calculation. If interpolating, this
    function uses linear interpolation. If another interpolation is desired,
    don't specify this parameter and perform desired interpolation on output.

    Args:
      G: A centered 2point statistic array. (n_sample, statistic dimensions)
      cutoff_r: Float. Return radii and probabilities beneath this value.
        Values greater than 1 are used as an exact radius cutoff. Values
        less than 1 are treated as a fraction of the maximum radius.
      interpolate_n: Int. Specify the number of equally spaced radii that
        the probabilities will be interpolated to.

    Returns:
      Pair Correlations Array (interpolate_n, n_samples+1). paircorr[:,0]
      is the radius values. paircorr[:,1:] are the probabilities, where
      paircorr[:,1] corresponds to the sample in G[0,...].



    '''
    faxes = lambda x: tuple(np.arange(x.ndim - 1) + 1)

    D = dist_from_center(G[0,...])

    D = D.reshape(-1)
    G = G.reshape(G.shape[0], -1)

    ind_sort = np.argsort(D)

    D = D[ind_sort]
    G = G[:, ind_sort]

    radii, ind_split = np.unique(D, return_index=True)

    probs = np.array([np.average(arr, axis=1) for arr in np.split(G, ind_split[1:], axis=1)])

    if cutoff_r is None:
        pass

    elif cutoff_r>1:
        r_inds = radii <= cutoff_r

        radii = radii[r_inds]
        probs = probs[r_inds, :]

    elif cutoff_r<1:
        r_inds = radii <= (cutoff_r*radii[-1])

        radii = radii[r_inds]
        probs = probs[r_inds, :]

    if interpolate_n:
        radii_out = np.linspace(radii.min(), radii.max(), interpolate_n)
        probs_out = np.zeros( (len(radii_out), probs.shape[1]) )

        for i in range(probs.shape[1]):
            probs_out[:,i] = np.interp(radii_out, radii, probs[:,i])

        return np.concatenate([radii_out.reshape(-1,1), probs_out], axis=1)
    else:
        return np.concatenate([radii.reshape(-1,1), probs], axis=1)
