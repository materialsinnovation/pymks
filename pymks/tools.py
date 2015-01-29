import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import itertools


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
    return colors.LinearSegmentedColormap('diff_cmap', cdict, 256)


def _get_spatial_correlations_cmap():
    HighRGB = np.array([118, 42, 131]) / 255.
    MediumRGB = np.array([255, 255, 191]) / 255.
    LowRGB = np.array([26, 152, 80]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('diff_cmap', cdict, 256)


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


def _get_coeff_cmap():
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
    coeff_cmap = _get_coeff_cmap()
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
        im = axs.imshow(microstructures[0].swapaxes(0, 1), cmap=plt.cm.gray,
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
    tmp = [[params['n_states'], -mean_score, scores.std()]
           for params, mean_score, scores in grid_scores]

    n_states, errors, stddev = list(zip(*tmp))
    plt.errorbar(n_states, errors, yerr=stddev, linewidth=2,
                 color=color, label=label)

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


def draw_PCA(X, n_sets):
    size = np.array(X.shape)
    if size[-1] == 2:
        _draw_PCA_2D(X, n_sets)
    elif size[-1] == 3:
        _draw_PCA_3D(X, n_sets)
    else:
        raise RuntimeError("n_components must be 2 or 3.")


def _get_PCA_color_list(n_sets):
    color_list = ['#1a9850', '#f46d43', '#762a83', '#1a1a1a',
                  '#ffffbf', '#a6d96a', '#c2a5cf', '#878787']
    return color_list[:n_sets]


def _draw_PCA_2D(X, n_sets):
    color_list = _get_PCA_color_list(n_sets)
    sets = np.array(X.shape)[0] / n_sets
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('PC 1', fontsize=15)
    ax.set_ylabel('PC 2', fontsize=15)
    ax.set_xticks(())
    ax.set_yticks(())
    for n in range(n_sets):
        ax.scatter(X[n * sets:(n + 1) * sets, 0],
                   X[n * sets:(n + 1) * sets, 1],
                   color=color_list[n])


def _draw_PCA_3D(X, n_sets):
    color_list = _get_PCA_color_list(n_sets)
    sets = np.array(X.shape)[0] / n_sets
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC 1', fontsize=10)
    ax.set_ylabel('PC 2', fontsize=10)
    ax.set_zlabel('PC 3', fontsize=10)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    for n in range(n_sets):
        ax.scatter(X[n * sets:(n + 1) * sets, 0],
                   X[n * sets:(n + 1) * sets, 1],
                   X[n * sets:(n + 1) * sets, 2],
                   color=color_list[n])


def draw_spatial_correlations(X_corr, correlation_plots=None):
    corr_cmap = _get_spatial_correlations_cmap()
    plt.close('all')
    vmin = np.min(X_corr)
    vmax = np.max(X_corr)
    n_corr = X_corr.shape[-1]
    n_states = ((np.sqrt(8 * n_corr + 1) - 1) / 2).astype(int)
    correlation_dict = _get_spatial_correlation_dict(X_corr, n_states)
    correlation_plot_names = _get_correlation_titles(correlation_dict,
                                                     correlation_plots)
    if correlation_plot_names is None:
        correlation_plot_names = correlation_dict.keys()
    n_plots = len(correlation_plot_names)
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))
    ii = 0
    for ax, title in zip(axs, correlation_plot_names):
        if ii == 0:
            im = ax.imshow(correlation_dict[title], cmap=corr_cmap,
                           interpolation='none', vmin=vmin, vmax=vmax)
        else:
            ax.imshow(correlation_dict[title], cmap=corr_cmap,
                      interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(r'Spatial Correlation $%s$' % title, fontsize=15)
        ii = ii + 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()


def _get_correlation_titles(correlation_dict, correlation_plots):
    if correlation_plots is None:
        return None
    correlation_plots = map(str, correlation_plots)
    new_names = correlation_plots
    for plot_name in correlation_plots:
        if plot_name not in correlation_dict:
            name = list(plot_name)
            name[1], name[4] = name[4], name[1]
            new_name = ''.join(str(e) for e in name)
            new_names[new_names.index(plot_name)] = new_name
            if new_name not in correlation_dict:
                raise RuntimeError("%s, correlation not found", plot_name)
    return new_names


def _get_spatial_correlation_dict(X_corr, n_states):
    keys = _get_spatial_correlation_titles(n_states)
    return dict(zip(keys, X_corr.swapaxes(0, -1)))


def _get_spatial_correlation_titles(n_states):
    auto = _get_autocorrelation_titles(n_states)
    cross = _get_crosscorrelation_titles(n_states)
    return map(str, auto + cross)


def _get_autocorrelation_titles(n_states):
    states = np.arange(n_states) + 1
    return list(zip(states, states))


def _get_crosscorrelation_titles(n_states):
    states = np.arange(n_states) + 1
    Niter = n_states / 2
    Nslice = n_states * (n_states - 1) / 2
    tmp = [zip(states, np.roll(states, i)) for i in range(1, Niter + 1)]
    titles = list(itertools.chain.from_iterable(tmp))
    return titles[:Nslice]


def draw_correlations(X_corr, correlations=None):
    n_states = ((np.sqrt(8 * X_corr.shape[-1] + 1) - 1) / 2).astype(int)
    X_auto_dict = _get_autocorrelation_dict(X_corr[..., :n_states])
    X_cross_dict = _get_crosscorrelation_dict(X_corr[..., n_states:])
    X_corr_dict = dict(X_cross_dict.items() + X_auto_dict.items())
    _draw_stats(X_corr_dict, correlations=correlations)


def draw_autocorrelations(X_auto, correlations=None):
    if X_auto.dtype == 'complex':
        print(DeprecationWarning("autocorrleation is complex."))
        X_auto = X_auto.real
    X_auto_dict = _get_autocorrelation_dict(X_auto)
    _draw_stats(X_auto_dict, correlations=correlations)


def draw_crosscorrelations(X_cross, correlations=None):
    if X_cross.dtype == 'complex':
        print(DeprecationWarning("crosscorrelation is complex"))
        X_cross = X_cross.real
    X_cross_dict = _get_crosscorrelation_dict(X_cross)
    _draw_stats(X_cross_dict, correlations=correlations)


def _get_autocorrelation_dict(X_auto):
    auto_labels = map(str, _get_autocorrelation_titles(X_auto.shape[-1]))
    return dict(zip(auto_labels, X_auto.swapaxes(0, -1)))


def _get_crosscorrelation_dict(X_cross):
    n_states = 0.5 + np.sqrt(1 + 8 * X_cross.shape[-1]) / 2.
    cross_labels = map(str, _get_crosscorrelation_titles(int(n_states)))
    return dict(zip(cross_labels, X_cross.swapaxes(0, -1)))


def _draw_stats(X_dict, correlations=None):
    X_cmap = _get_coeff_cmap()
    plt.close('all')
    correlation_labels = _get_correlation_titles(X_dict, correlations)
    if correlation_labels is None:
        correlation_labels = X_dict.keys()
    n_plots = len(correlation_labels)
    X_list = [v[..., None]
              for k, v in X_dict.items() if k in correlation_labels]
    X_ = np.concatenate(tuple(X_list), axis=-1)
    vmin = np.min(X_)
    vmax = np.max(X_)
    x_loc, x_labels = _get_ticks_params(X_.shape[0])
    y_loc, y_labels = _get_ticks_params(X_.shape[1])
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))
    if n_plots == 1:
        axs = list([axs])
    ii = 0
    for ax, label in zip(axs, correlation_labels):
        ax.set_xticks(x_loc)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.set_yticks(y_loc)
        ax.set_yticklabels(y_labels, fontsize=12)
        im = ax.imshow(X_dict[label], cmap=X_cmap,
                       interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_title(r"Correlation $h = {0}$, $h' = {1}$".format(label[1],
                                                                label[-2]),
                     fontsize=15)
        fig.subplots_adjust(right=0.8)
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="10%", pad=0.05)
        cbar_ticks = _get_colorbar_ticks(X_dict[label])
        cbar = plt.colorbar(im, cax=cbar_ax, ticks=cbar_ticks,
                            boundaries=np.arange(cbar_ticks[0],
                                                 cbar_ticks[-1] + 0.005,
                                                 0.005))
        cbar.ax.tick_params(labelsize=12)
        ii = ii + 1
        fig.subplots_adjust(right=0.8)
        plt.tight_layout()


def _get_ticks_params(X):
    segments = np.roll(np.arange(4, 7, dtype=int), 1, 0)
    m = segments[np.argmin(X % segments)]
    n = max((X + 1) / m, 1)
    tick_loc = range(0, X + n, n)
    tick_labels = range(- (X - 1) / 2, (X + 1) / 2 + n, n)
    return tick_loc, tick_labels


def _get_colorbar_ticks(X_):
    tick_range = np.linspace(np.min(X_), np.max(X_), 5)
    return tick_range.astype(float)


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
