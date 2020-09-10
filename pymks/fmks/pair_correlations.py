
def dist_from_center(C):
    '''
    Calculate the distance from the center pixel of a
    centered array (like 2point statistics).
    Supports any number of dimensions.

    Args
    ----

    C: Centered array. Must support ndim and
    shape methods (like numpy arrays).
    '''

    args = [
        np.linspace(
            -(C.shape[C.ndim-i]//2),
            C.shape[C.ndim-i]//2 + C.shape[C.ndim-i]%2 - 1,
            C.shape[C.ndim-i]
        )
        for i in range(1, C.ndim+1)
    ]

    D = sum([a**2 for a in np.meshgrid(*args)])

    return np.sqrt(D)


def paircorr_from_twopoint(G, interpolate_n=None):
    '''
    Computes the pair correlations from 2point statistics.

    Args
    ----

    G: A centered 2point statistic array.
    interpolate: Should the pair correlation be interpolated such that
        probabilities are reported for equally spaced radii? If so, enter
        the number of radii you want returned. Uses linear interpolation,
        if another interpolation is desired, don't specify this parameter
        and perform desired interpolation on output.
    '''
    D = dist_from_center(G)

    D = D.ravel()
    G = G.ravel()

    ind_sort = np.argsort(D)

    D = D[ind_sort]
    G = G[ind_sort]

    radii, ind_split = np.unique(D, return_index=True)

    probs = np.array([np.average(arr) for arr in np.split(G, ind_split[1:])])

    if interpolate_n:
        radii_out = np.linspace(radii.min(), radii.max(), interpolate_n)
        probs_out = np.interp(radii_out, radii, probs)

        return np.stack([radii_out, probs_out], axis=1)
    else:
        return np.stack([radii, probs], axis=1)
