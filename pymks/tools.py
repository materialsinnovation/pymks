import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np


def _set_colors():
    HighRGB = np.array([26, 152, 80]) / 255.
    MediumRGB = np.array([255, 255, 191]) / 255.
    LowRGB = np.array([0, 0, 0]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    plt.register_cmap(name='PyMKS', data=cdict)
    plt.set_cmap('PyMKS')


def _get_response_cmap():
    HighRGB = np.array([26, 152, 80]) / 255.
    MediumRGB = np.array([255, 255, 191]) / 255.
    LowRGB = np.array([0, 0, 0]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('coeff_cmap', cdict, 256)


def _get_diff_cmap():
    HighRGB = np.array([118, 42, 131]) / 255.
    MediumRGB = np.array([255, 255, 191]) / 255.
    LowRGB = np.array([0, 0, 0]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('coeff_cmap', cdict, 256)


def _set_cdict(HighRGB, MediumRGB, LowRGB):
    cdict = {'red': ((0.0, LowRGB[0], LowRGB[0]),
                    (0.5, MediumRGB[0], MediumRGB[0]),
                    (1.0, HighRGB[0], HighRGB[0])),

             'green': ((0.0, LowRGB[1], LowRGB[1]),
                       (0.5, MediumRGB[1], MediumRGB[1]),
                       (1.0, HighRGB[1], HighRGB[1])),

             'blue': ((0.0, LowRGB[2], LowRGB[2]),
                      (0.5, MediumRGB[2], MediumRGB[2]),
                      (1.0, HighRGB[2], HighRGB[2]))}

    return cdict


def _getCoeffCmap():
    HighRGB = np.array([244, 109, 67]) / 255.
    MediumRGB = np.array([255, 255, 191]) / 255.
    LowRGB = np.array([0, 0, 0]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('coeff_cmap', cdict, 256)


def draw_microstructure_discretization(M, a=0, s=0, Nbin=6,
                                       bound=0.016, height=1.7, ax=None):
    r""" Creates a diagram to illustrate the binning of a continues values
    in local state space.

    Args:
        Array representing a microstructure with a continuous variable.
    Returns:
        Image of the continuous local state binned discretely in the local
        state space.
    """
    if ax is not None:
        ax = plt.axes()
    dx = 1. / (Nbin - 1.)

    cm = plt.get_cmap('cubehelix')
    cNorm = colors.Normalize(vmin=0, vmax=Nbin - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for i in range(Nbin - 1):
        color = scalarMap.to_rgba(i)
        r = plt.Rectangle((i * dx, 0), dx, dx, lw=4, ec='k', color=color)
        ax.add_patch(r)

    plt.yticks(())

    plt.ylim(ymax=dx * height, ymin=-bound)
    plt.xlim(xmin=-bound, xmax=1 + bound)

    ax.set_aspect('equal')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    for line in ax.xaxis.get_ticklines():
        line.set_markersize(0)

    all_spines = ['top', 'bottom', 'right', 'left']
    for spline in all_spines:
        ax.spines[spline].set_visible(False)

    plt.xlabel(r'$\chi^h \;\; \left( H = 6 \right)$', fontsize=16)

    v = M[a, s]

    H = np.linspace(0, 1, Nbin)
    m = np.maximum(1 - abs(v - H) / dx, 0)
    Mstring = r'$m_{{{a},{s}}}={v:1.2g}$'.format(a=a, s=s, v=v)
    arr = r'{0:1.2g}'.format(m[0])
    for i in range(1, len(m)):
        arr += r', {0:1.2g}'.format(m[i])
    mstring = r'$m_{{{a},{s}}}^h=\left({arr}\right)$'.format(a=a, s=s, arr=arr)

    plt.plot((v, v), (0, dx * height), 'r--', lw=3)
    plt.text(v + 0.02,
             dx * (1 + 0.65 * (height - 1)),
             Mstring,
             fontsize=16,
             color='r')
    plt.text(v + 0.02,
             dx * (1 + 0.2 * (height - 1)),
             mstring,
             fontsize=16,
             color='r')


def draw_coeff(coeff):
    if coeff.dtype == 'complex':
        print(DeprecationWarning("Coefficients are complex."))
        coeff = coeff.real
    coeff_cmap = _getCoeffCmap()
    plt.close('all')
    vmin = np.min(coeff)
    vmax = np.max(coeff)
    Ncoeff = coeff.shape[-1]
    fig, axs = plt.subplots(1, Ncoeff, figsize=(Ncoeff * 4, 4))
    ii = 0
    for ax in axs:
        if ii == 0:
            im = ax.imshow(coeff[..., ii].swapaxes(0, 1), cmap=coeff_cmap,
                           interpolation='none', vmin=vmin, vmax=vmax)
        else:
            ax.imshow(coeff[..., ii].swapaxes(0, 1), cmap=coeff_cmap,
                      interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(r'Influence Coefficients $h = %s$' % ii, fontsize=15)
        ii = ii + 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()


def draw_microstructure_strain(microstructure, strain):
    plt.close('all')
    cmap = _get_response_cmap()
    fig = plt.figure(figsize=(8, 4))
    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(microstructure.swapaxes(0, 1), cmap=plt.cm.gray,
               interpolation='none')
    ax0.set_xticks(())
    ax0.set_yticks(())
    ax1 = plt.subplot(1, 2, 2)
    im1 = ax1.imshow(strain.swapaxes(0, 1), cmap=cmap, interpolation='none')
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title(r'$\mathbf{\varepsilon_{xx}}$', fontsize=25)
    ax0.set_title('Microstructure', fontsize=20)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im1, cax=cbar_ax)

    plt.tight_layout()


def draw_microstructures(*microstructures):
    n_micros = len(microstructures)
    vmin = np.min(microstructures)
    vmax = np.max(microstructures)
    plt.close('all')
    fig, axs = plt.subplots(1, n_micros, figsize=(n_micros * 4, 4))
    if n_micros > 1:
        for micro, ax in zip(microstructures, axs.flat):
            im = ax.imshow(micro.swapaxes(0, 1), cmap=plt.cm.gray,
                           interpolation='none', vmin=vmin, vmax=vmax)
            ax.set_xticks(())
            ax.set_yticks(())
    else:
        im = axs.imshow(microstructures.swapaxes(0, 1), cmap=plt.cm.gray,
                        interpolation='none', vmin=vmin, vmax=vmax)
        axs.set_xticks(())
        axs.set_yticks(())
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()


def draw_strains(*strains, **titles):
    n_strains = len(strains)
    plt.close('all')
    cmap = _get_response_cmap()
    fig, axs = plt.subplots(1, n_strains, figsize=(n_strains * 4, 4))
    if n_strains > 1:
        for micro, ax, title in zip(strains, axs, titles):
            im = ax.imshow(micro.swapaxes(0, 1), cmap=cmap,
                           interpolation='none')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(r'$\mathbf{\varepsilon_{%s}}$' % titles[title],
                         fontsize=25)
    else:
        micro = np.array(strains)[0]
        im = axs.imshow(micro.swapaxes(0, 1), cmap=cmap, interpolation='none')
        axs.set_xticks(())
        axs.set_yticks(())
        axs.set_title(r'$\mathbf{\varepsilon_{%s}}$'
                      % next(iter(titles.values())), fontsize=25)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()


def draw_concentrations(*concentrations, **titles):
    n_concens = len(concentrations)
    vmin = np.min(concentrations)
    vmax = np.max(concentrations)
    cmap = _get_response_cmap()
    plt.close('all')
    fig, axs = plt.subplots(1, n_concens, figsize=(n_concens * 4, 4))
    if n_concens > 1:
        for concen, ax, title in zip(concentrations, axs, titles):
            im = ax.imshow(concen.swapaxes(0, 1), cmap=cmap,
                           interpolation='none', vmin=vmin, vmax=vmax)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title('Concentration (%s)' % titles[title],
                         fontsize=15)
    else:
        im = axs.imshow(concentrations[0].swapaxes(0, 1), cmap=cmap,
                        interpolation='none', vmin=vmin, vmax=vmax)
        axs.set_xticks(())
        axs.set_yticks(())
        axs.set_title('Concentration (%s)' % next(iter(titles.values())),
                      fontsize=15)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()


def draw_strains_compare(strain1, strain2):
    plt.close('all')
    cmap = _get_response_cmap()
    vmin = min((strain1.flatten().min(), strain2.flatten().min()))
    vmax = max((strain1.flatten().max(), strain2.flatten().max()))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    titles = ['Finite Element', 'MKS']
    strains = (strain1, strain2)
    for strain, ax, title in zip(strains, axs, titles):
        im = ax.imshow(strain.swapaxes(0, 1), cmap=cmap,
                       interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(r'$\mathbf{\varepsilon_{xx}}$ (%s)' % title, fontsize=20)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout()


def draw_concentrations_compare(con1, con2):
    plt.close('all')
    cmap = _get_response_cmap()
    vmin = min((con1.flatten().min(), con2.flatten().min()))
    vmax = max((con1.flatten().max(), con2.flatten().max()))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    titles = ['Simulation', 'MKS']
    cons = (con1, con2)
    for con, ax, title in zip(cons, axs, titles):
        im = ax.imshow(con.swapaxes(0, 1), cmap=cmap, interpolation='none',
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('Concentration (%s)' % title, fontsize=15)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout()


def draw_diff(*responses, **titles):
    n_responses = len(responses)
    vmin = np.min(responses)
    vmax = np.max(responses)
    cmap = _get_diff_cmap()
    plt.close('all')
    fig, axs = plt.subplots(1, n_responses, figsize=(n_responses * 4, 4))
    if n_responses > 1:
        for response, ax, title in zip(responses, axs, titles):
            im = ax.imshow(response.swapaxes(0, 1), cmap=cmap,
                           interpolation='none', vmin=vmin, vmax=vmax)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title('%s' % titles[title], fontsize=15)
    else:
        im = axs.imshow(responses[0].swapaxes(0, 1), cmap=cmap,
                        interpolation='none', vmin=vmin, vmax=vmax)
        axs.set_xticks(())
        axs.set_yticks(())
        axs.set_title('%s' % next(iter(titles.values())), fontsize=15)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()


def draw_gridscores(grid_scores, label=None, color='#f46d43'):
    tmp = [[params['n_states'], -mean_score, scores.std()] \
            for params, mean_score, scores in grid_scores]
    
    n_states, errors, stddev = list(zip(*tmp))
    plt.errorbar(n_states, errors, yerr=stddev, linewidth=2, color=color, label=label)

    plt.legend()
    plt.ylabel('MSE', fontsize=20)
    plt.xlabel('Number of Local States', fontsize=15)


def bin(arr, n_bins):
    r"""
    Discretize the array `arr`, which must be between 0 and 1.

    >>> res = bin(np.array((0.2, 0.5, 0.7)), 4)
    >>> np.allclose(res,
    ...             [[ 0.4,  0.6,  0. ,  0. ],
    ...              [ 0. ,  0.5,  0.5,  0. ],
    ...              [ 0. ,  0. ,  0.9,  0.1]])
    True

    Args:
        arr: Array that must be between 0 and 1.
        n_bins: Integer value representing the number of local states
             in the local state space of the microstructure function.
    Returns:
        Microstructure function for array `arr`.
    """
    X = np.linspace(0, 1, n_bins)
    dX = X[1] - X[0]

    return np.maximum(1 - abs(arr[:, None] - X) / dX, 0)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Code taken from scikit-learn examples for version 0.15.

    Generate a simple plot of the test and traning learning curve.

    Args:
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.
        title: string
            Used for the title for the chart.
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y: array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.
        cv : integer, cross-validation generator, optional
            If an integer is passed, it is the number of folds (defaults to 3).
            Specific cross-validation objects can be passed, see
            sklearn.cross_validation module for the list of possible objects
        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used
            to generate the learning curve. If the dtype is float, it is
            regarded as a fraction of the maximum size of the training set
            (that is determined by the selected validation method), i.e. it has
            to be within (0, 1]. Otherwise it is interpreted as absolute sizes
            of the training sets. Note that for classification the number of
            samples usually have to be big enough to contain at least one
            sample from each class. (default: np.linspace(0.1, 1.0, 5))

        Returns:
            A plot of the learning curves for both the training curve and the
            cross-validation curve.
    """
    from sklearn.learning_curve import learning_curve

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
