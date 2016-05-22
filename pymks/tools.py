try:
    import matplotlib.pyplot as plt
except ImportError:
    import pytest
    pytest.importorskip('matplotlib')
    raise
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.learning_curve import learning_curve
from .stats import _auto_correlations
from .stats import _cross_correlations
import numpy as np
import warnings

warnings.filterwarnings("ignore")
plt.style.library['ggplot']['text.color'] = '#555555'
plt.style.use('ggplot')


def _set_colors():
    """
    Helper function used to set the color map.
    """
    HighRGB = np.array([26, 152, 80]) / 255.
    MediumRGB = np.array([255, 255, 191]) / 255.
    LowRGB = np.array([0, 0, 0]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    plt.register_cmap(name='PyMKS', data=cdict)
    plt.set_cmap('PyMKS')


def _get_response_cmap():
    """
    Helper function used to set the response color map.

    Returns:
        dictionary with colors and localizations on color bar.
    """
    HighRGB = np.array([179, 255, 204]) / 255.
    MediumRGB = np.array([28, 137, 63]) / 255.
    LowRGB = np.array([11, 53, 24]) / 255.

    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('coeff_cmap', cdict, 256)


def _get_microstructure_cmap():
    """
    Helper function used to set the microstructure color map.

    Returns:
        dictionary with colors and microstructure on color bar.
    """

    HighRGB = np.array([229, 229, 229]) / 255.
    MediumRGB = np.array([114.5, 114.5, 114.5]) / 255.
    LowRGB = np.array([0, 0, 0]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('micro_cmap', cdict, 256)


def _get_diff_cmap():
    """
    Helper function used to set the difference color map.

    Returns:
        dictionary with colors and localizations on color bar.
    """
    HighRGB = np.array([255, 207, 181]) / 255.
    MediumRGB = np.array([238, 86, 52]) / 255.
    LowRGB = np.array([99, 35, 21]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('diff_cmap', cdict, 256)


def _grid_matrix_cmap():
    """
    Helper function used to set the grid matrix color map.

    Returns:
        dictionary with colors and localizations on color bar.
    """
    HighRGB = np.array([229, 229, 229]) / 255.
    MediumRGB = np.array([114.5, 114.5, 114.5]) / 255.
    LowRGB = np.array([0, 0, 0]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('grid_cmap', cdict, 256)


def _set_cdict(HighRGB, MediumRGB, LowRGB):
    """
    Helper function used to set color map from 3 RGB values.

    Args:
        HighRGB: RGB with highest values
        MediumRGB: RGB with medium values
        LowRGB: RGB with lowest values

    Returns:
        dictionary with colors and localizations on color bar.
    """
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
    """
    Helper function used to set the influence coefficients color map.

    Returns
    """
    HighRGB = np.array([205, 0, 29]) / 255.
    MediumRGB = np.array([240, 240, 240]) / 255.
    LowRGB = np.array([17, 55, 126]) / 255.
    cdict = _set_cdict(HighRGB, MediumRGB, LowRGB)
    return colors.LinearSegmentedColormap('coeff_cmap', cdict, 256)


def _get_color_list(n_sets):
    """
    color list for dimensionality reduction plots

    Args:
        n_sets: number of dataset

    Returns:
        list of colors for n_sets
    """
    color_list = ['#1a9850', '#f46d43', '#1f78b4', '#e31a1c',
                  '#6a3d9a', '#b2df8a', '#fdbf6f', '#a6cee3',
                  '#fb9a99', '#cab2d6', '#ffff99', '#b15928']

    return color_list[:n_sets]


def draw_coeff(coeff, fontsize=15, figsize=None):
    """
    Visualize influence coefficients.

    Args:
        coeff (ND array): influence coefficients with dimensions (x, y,
            n_states)
        fontsize (int, optional): values used for the title font size
    """
    plt.close('all')
    coeff_cmap = _get_coeff_cmap()
    n_coeff = coeff.shape[-1]
    titles = [r'Influence Coefficients $l = %s$' % ii for ii
              in np.arange(n_coeff)]
    _draw_fields(np.rollaxis(coeff, -1, 0), coeff_cmap,
                 fontsize=fontsize, titles=titles, figsize=figsize)


def draw_microstructure_strain(microstructure, strain):
    """
    Draw microstructure and its associated strain

    Args:
        microstructure (2D array): numpy array with dimensions (x, y)
        strain (2D array): numpy array with dimensions (x, y)
    """
    plt.close('all')
    cmap = _get_response_cmap()
    fig = plt.figure(figsize=(8, 4))
    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(microstructure, cmap=_get_microstructure_cmap(),
               interpolation='none')
    ax0.set_xticks(())
    ax0.set_yticks(())
    ax1 = plt.subplot(1, 2, 2)
    im1 = ax1.imshow(strain, cmap=cmap, interpolation='none')
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title(r'$\mathbf{\varepsilon_{xx}}$', fontsize=25)
    ax0.set_title('Microstructure', fontsize=20)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    fig.colorbar(im1, cax=cbar_ax)
    plt.tight_layout()
    plt.show()


def draw_microstructures(*microstructures):
    """
    Draw microstructures

    Args:
        microstructures (3D array): numpy array with dimensions
            (n_samples, x, y)
    """
    cmap = _get_microstructure_cmap()
    titles = [' ' for s in np.arange(microstructures[0].shape[0])]
    _draw_fields(microstructures[0], cmap, 10, titles)


def draw_strains(strains, labels=None, fontsize=15):
    """
    Draw strain fields

    Args:
        strains (3D array): numpy arrays with dimensions (n_samples, x, y)
        labels (list, str, optional): titles for strain fields
        fontsize (int, optional): title font size
    """
    cmap = _get_response_cmap()
    if labels is None:
        labels = [' ' for s in strains]
    _draw_fields(strains, cmap, fontsize, labels)


def draw_concentrations(concentrations, labels=None, fontsize=15):
    """Draw comparison fields

    Args:
        concentrations (list): numpy arrays with dimensions (x, y)
        labels (list): titles for concentrations
        fontsize (int): used for the title font size
    """
    if labels is None:
        labels = [" " for s in concentrations]
    cmap = _get_response_cmap()
    _draw_fields(concentrations, cmap, fontsize, labels)


def draw_strains_compare(strain_FEM, strain_MKS, fontsize=20):
    """Draw comparison of strain fields.

    Args:
        strain_FEM (2D array): strain field with dimensions (x, y) from finite
            element
        strain_MKS (2D array): strain fieldwith dimensions (x, y) from MKS
        fontsize (int, optional): scalar values used for the title font size
    """
    cmap = _get_response_cmap()
    titles = ['Finite Element', 'MKS']
    titles_ = [r'$\mathbf{\varepsilon_{xx}}$ - %s' % title for title in titles]
    _draw_fields((strain_FEM, strain_MKS), cmap, fontsize, titles_)


def draw_concentrations_compare(concentrations, labels, fontsize=15):
    """Draw comparesion of concentrations.

    Args:
        concentrations (3D array): list of difference arrays with dimensions
            (x, y)
        labels (list, str): list of titles for difference arrays
        fontsize (int, optional): scalar values used for the title font size
    """
    cmap = _get_response_cmap()
    _draw_fields(concentrations, cmap, fontsize, labels)


def draw_differences(differences, labels=None, fontsize=15):
    """Draw differences in predicted response fields.

    Args:
        differences (list, 2D arrays): list of difference arrays with
            dimesions (x, y).
        labels (list, str, optional): titles for difference arrays
        fontsize (int, optional): scalar values used for the title font size
    """
    cmap = _get_diff_cmap()
    if labels is None:
        labels = [' ' for s in differences]
    _draw_fields(differences, cmap, fontsize, labels)


def _draw_fields(fields, field_cmap, fontsize, titles, figsize=None):
    """
    Helper function used to draw fields.

    Args:
        fields - iterable object with 2D numpy arrays
        field_cmap - color map for plot
        fontsize - font size for titles and color bar text
        titles - titles for plot
    """
    plt.close('all')
    vmin = np.min(fields)
    vmax = np.max(fields)
    n_fields = len(fields)
    if titles is not None:
        n_titles = len(titles)
        if n_fields != n_titles:
            raise RuntimeError(
                "number of plots does not match number of labels.")
    plt.close('all')
    if figsize is None:
        figsize = (1, n_fields)
    fig, axs = plt.subplots(figsize[0], figsize[1],
                            figsize=(figsize[1] * 4, figsize[0] * 4))

    if n_fields > 1:
        for field, ax, title in zip(fields, axs.flat, titles):
            im = ax.imshow(field,
                           cmap=field_cmap, interpolation='none',
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title, fontsize=fontsize)
    else:
        im = axs.imshow(fields[0], cmap=field_cmap,
                        interpolation='none', vmin=vmin, vmax=vmax)
        axs.set_xticks(())
        axs.set_yticks(())
        axs.set_title(titles[0], fontsize=fontsize)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
    cbar_font = np.floor(0.8 * fontsize)
    cbar_ax.tick_params(labelsize=cbar_font)
    cbar_ax.yaxis.set_offset_position('right')
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()
    plt.rc('font', **{'size': str(cbar_font)})
    plt.show()


def draw_gridscores(grid_scores, param, score_label=None, colors=None,
                    data_labels=None, param_label=None, fontsize=20):
    """
    Visualize the score values and standard deviations from grids
    scores result from GridSearchCV while varying 1 parameters.

    Args:
        grid_scores (list, grid_scores): `grid_scores_` attribute from
            GridSearchCV
        param (list, str): parameters used in grid_scores
        score_label (str): label for score value axis
        colors (list): colors used for this specified parameter
        param_label (list): parameter titles to appear on plot
    """
    plt.close('all')
    if type(grid_scores[0]) is not list:
        grid_scores = [grid_scores]
    if data_labels is None:
        data_labels = [None for l in range(len(grid_scores))]
    if score_label is None:
        score_label = ''
    if param_label is None:
        param_label is ''
    if colors is None:
        colors = _get_color_list(len(grid_scores))
    if len(grid_scores) != len(data_labels) or len(data_labels) != len(colors):
        raise RuntimeError(
            "grid_scores, colors, and param_lables must have the same length.")
    mins, maxes = [], []
    for grid_score, data_label, color in zip(grid_scores, data_labels, colors):
        tmp = [[params[param], mean_score, scores.std()]
               for params, mean_score, scores in grid_score]
        _param, errors, stddev = list(zip(*tmp))
        _mins = np.array(errors) - np.array(stddev)
        _maxes = np.array(errors) + np.array(stddev)
        plt.fill_between(_param, _mins, _maxes, alpha=0.1,
                         color=color)
        plt.plot(_param, errors, 'o-', color=color, label=data_label,
                 linewidth=2)
        mins.append(min(_mins))
        maxes.append(max(_maxes))
    if data_labels[0] is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   fontsize=15)
    _min, _max = min(mins), max(maxes)
    y_epsilon = (_max - _min) * 0.05
    plt.ylim((_min - y_epsilon, _max + y_epsilon))
    plt.ticklabel_format(style='sci', axis='y')
    plt.ylabel(score_label, fontsize=fontsize)
    plt.xlabel(param_label, fontsize=fontsize)
    plt.show()


def draw_gridscores_matrix(grid_scores, params, score_label=None,
                           param_labels=None):
    """
    Visualize the score value matrix and standard deviation matrix from grids
    scores result from GridSearchCV while varying two parameters.

    Args:
        grid_scores (list): `grid_scores_` attribute from GridSearchCV
        params (list): two parameters used in grid_scores
        score_label (str): label for score value axis
        param_labels (list): parameter titles to appear on plot
    """
    plt.close('all')
    if score_label is None:
        score_label = 'R-Squared'
    if param_labels is None:
        param_labels = ['', '']
    tmp = [[params, mean_score, scores.std()]
           for parameters, mean_score, scores in grid_scores.grid_scores_]
    param, means, stddev = list(zip(*tmp))
    param_range_0 = grid_scores.param_grid[params[0]]
    param_range_1 = grid_scores.param_grid[params[1]]
    mat_size = (len(param_range_1), len(param_range_0))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    matrices = np.concatenate((np.array(means).reshape(mat_size)[None],
                               np.array(stddev).reshape(mat_size)[None]))
    X_cmap = _grid_matrix_cmap()
    x_label = param_labels[0]
    y_label = param_labels[1]
    plot_title = [score_label, 'Standard Deviation']
    for ax, label, matrix, title in zip(axs, param_labels,
                                        matrices,
                                        plot_title):
        ax.set_xticklabels(param_range_0, fontsize=12)
        ax.set_yticklabels(param_range_1, fontsize=12)
        ax.set_xticks(np.arange(len(param_range_0)))
        ax.set_yticks(np.arange(len(param_range_0)))
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(False)
        im = ax.imshow(matrix,
                       cmap=X_cmap, interpolation='none')
        ax.set_title(title, fontsize=22)
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="10%", pad=0.05)
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        fig.subplots_adjust(right=1.2)
    plt.show()


def draw_component_variance(variance):
    """
    Visualize the percent variance as a function of components.

    Args:
        variance (list): variance ratio explanation from dimensional
            reduction technique.
    """
    plt.close('all')
    n_components = len(variance)
    x = np.arange(1, n_components + 1)
    plt.plot(x, np.cumsum(variance * 100), 'o-', color='#1a9641', linewidth=2)
    plt.xlabel('Number of Components', fontsize=15)
    plt.xlim(0, n_components + 1)
    plt.ylabel('Percent Variance', fontsize=15)
    plt.show()


def draw_components_scatter(datasets, labels, title=None,
                            component_labels=None, view_angles=None,
                            legend_outside=False, fig_size=None):
    """
    Visualize low dimensional representations of microstructures.

    Args:
        datasets (list, 2D arrays): low dimensional data with dimensions
            [n_samples, n_components]. The length of n_components must be 2 or
            3.
        labels (list, str): list of lables for each of each array datasets
        title: main title for plot
        component_labels: labels for components
        view_angles (int,int): the elevation and azimuth angles of the axes
            to rotate the axes.
        legend_outside: specify to move legend box outside the main plot
            domain
        figsize: (width, height) figure size in inches
    """
    plt.close('all')
    if title is None:
        title = 'Low Dimensional Representation'
    n_components = np.array(datasets[0][-1].shape)
    if component_labels is None:
        component_labels = range(1, n_components + 1)
    if len(datasets) != len(labels):
        raise RuntimeError('datasets and labels must have the same length')
    if n_components != len(component_labels):
        raise RuntimeError('number of components and component_labels must'
                           ' have the same length')
    if n_components[-1] == 2:
        _draw_components_2D(datasets, labels, title, component_labels[:2],
                            legend_outside, fig_size)
    elif n_components[-1] == 3:
        _draw_components_3D(datasets, labels, title, component_labels,
                            view_angles, legend_outside, fig_size)
    else:
        raise RuntimeError("n_components must be 2 or 3.")


def draw_evolution(datasets, labels, title=None, component_labels=None,
                   view_angles=None, legend_outside=False, fig_size=None):
    """
    Visualize low dimensional representations of microstructures.

    Args:
        datasets (list, 2D arrays): low dimensional data with dimensions
            [n_samples, n_components]. The length of n_components must be 2 or
            3.
        labels (list, str): list of lables for each of each array datasets
        title: main title for plot
        component_labels: labels for components
        view_angles (int,int): the elevation and azimuth angles of the axes
            to rotate the axes.
        legend_outside: specify to move legend box outside the main plot
            domain
        figsize: (width, height) figure size in inches
    """
    plt.close('all')
    if title is None:
        title = 'Low Dimensional Representation'
    n_components = np.array(datasets[0][-1].shape)
    if component_labels is None:
        component_labels = range(1, n_components + 1)
    if len(datasets) != len(labels):
        raise RuntimeError('datasets and labels must have the same length')
    if n_components != len(component_labels):
        raise RuntimeError('number of components and component_labels must'
                           ' have the same length')
    if n_components[-1] == 2:
        _draw_components_evolution(datasets, labels,
                                   title, component_labels[:2],
                                   legend_outside, fig_size)
    else:
        raise RuntimeError("time and one component must be paired")


def _draw_components_2D(X, labels, title, component_labels,
                        legend_outside, fig_size):
    """
    Helper function to plot 2 components.

    Args:
        X: Arrays with low dimensional data
        labels: labels for each of the low dimensional arrays
    """
    n_sets = len(X)
    color_list = _get_color_list(n_sets)
    if fig_size is not None:
        fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Component ' + str(component_labels[0]), fontsize=15)
    ax.set_ylabel('Component ' + str(component_labels[1]), fontsize=15)
    X_array = np.concatenate(X)
    x_min, x_max = [np.min(X_array[:, 0]), np.max(X_array[:, 0])]
    y_min, y_max = [np.min(X_array[:, 1]), np.max(X_array[:, 1])]
    x_epsilon = (x_max - x_min) * 0.05
    y_epsilon = (y_max - y_min) * 0.05
    ax.set_xlim([x_min - x_epsilon, x_max + x_epsilon])
    ax.set_ylim([y_min - y_epsilon, y_max + y_epsilon])
    for label, pts, color in zip(labels, X, color_list):
        ax.plot(pts[:, 0], pts[:, 1], 'o', color=color, label=label)
        lg = plt.legend(loc=1, borderaxespad=0., fontsize=15)
    if legend_outside is not False:
        lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2,
                        borderaxespad=0., fontsize=15)
    lg.draggable()
    plt.title(title, fontsize=20)
    plt.show()


def _draw_components_evolution(X, labels, title, component_labels,
                               legend_outside, fig_size):
    """
    Helper function to plot 2 components.

    Args:
        X: Arrays with low dimensional data
        labels: labels for each of the low dimensional arrays
    """
    n_sets = len(X)
    color_list = _get_color_list(n_sets)
    if fig_size is not None:
        fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel('Components ', fontsize=15)
    X_array = np.concatenate(X)
    x_min, x_max = [np.min(X_array[:, 0]), np.max(X_array[:, 0])]
    y_min, y_max = [np.min(X_array[:, 1]), np.max(X_array[:, 1])]
    x_epsilon = (x_max - x_min) * 0.05
    y_epsilon = (y_max - y_min) * 0.05
    ax.set_xlim([x_min - x_epsilon, x_max + x_epsilon])
    ax.set_ylim([y_min - y_epsilon, y_max + y_epsilon])
    for label, pts, color in zip(labels, X, color_list):
        ax.plot(pts[:, 0], pts[:, 1], 'o', color=color, label=label)
        lg = plt.legend(loc=1, borderaxespad=0., fontsize=15)
    if legend_outside:
        lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2,
                        borderaxespad=0., fontsize=15)
    lg.draggable()
    plt.title(title, fontsize=20)
    plt.show()


def _draw_components_3D(X, labels, title, component_labels, view_angles,
                        legend_outside, fig_size):
    """
    Helper function to plot 2 components.

    Args:
        X: Arrays with low dimensional data
        labels: labels for each of the low dimensional arrays
    """
    n_sets = len(X)
    color_list = _get_color_list(n_sets)
    if fig_size is not None:
        fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Component ' + str(component_labels[0]), fontsize=12)
    ax.set_ylabel('Component ' + str(component_labels[1]), fontsize=12)
    ax.set_zlabel('Component ' + str(component_labels[2]), fontsize=12)
    X_array = np.concatenate(X)
    x_min, x_max = [np.min(X_array[:, 0]), np.max(X_array[:, 0])]
    y_min, y_max = [np.min(X_array[:, 1]), np.max(X_array[:, 1])]
    z_min, z_max = [np.min(X_array[:, 2]), np.max(X_array[:, 2])]
    x_epsilon = (x_max - x_min) * 0.05
    y_epsilon = (y_max - y_min) * 0.05
    z_epsilon = (z_max - z_min) * 0.05
    ax.set_xlim([x_min - x_epsilon, x_max + x_epsilon])
    ax.set_ylim([y_min - y_epsilon, y_max + y_epsilon])
    ax.set_zlim([z_min - z_epsilon, z_max + z_epsilon])
    for label, pts, color in zip(labels, X, color_list):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o', color=color, label=label)
    plt.title(title, fontsize=15)
    if view_angles is not None:
        ax.view_init(view_angles[0], view_angles[1])
    lg = plt.legend(loc=1, borderaxespad=0., fontsize=15)
    if legend_outside:
        lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2,
                        borderaxespad=0., fontsize=15)
    plt.show()


def draw_goodness_of_fit(fit_data, pred_data, labels):
    """Goodness of fit plot for MKSHomogenizationModel.

    Args:
        fit_data (2D array): Low dimensional representation of the prediction
            values of the data used to fit the model and the actual values.
        pred_data (2D array): Low dimensional representation of the prediction
            values of the data used for prediction with the model and the
            actual values.
    """
    plt.close('all')
    y_total = np.concatenate((fit_data, pred_data), axis=-1)
    y_min, y_max = np.min(y_total), np.max(y_total)
    middle = (y_max + y_min) / 2.
    data_range = y_max - y_min
    line = np.linspace(middle - data_range * 1.03 / 2,
                       middle + data_range * 1.03 / 2, endpoint=False)
    plt.plot(line, line, '-', linewidth=3, color='#000000')
    plt.plot(fit_data[0], fit_data[1], 'o', color='#1a9850', label=labels[0])
    plt.plot(pred_data[0], pred_data[1], 'o',
             color='#f46d43', label=labels[1])
    plt.title('Goodness of Fit', fontsize=20)
    plt.xlabel('Actual', fontsize=18)
    plt.ylabel('Predicted', fontsize=18)
    plt.legend(loc=2, fontsize=15)
    plt.show()


def draw_components(X_comp, fontsize=15, figsize=None):
    """
    Visualize spatial correlations.

    Args:
        X_corr (ND array): correlations
        correlations (list, optional): correlation labels
    """
    cmap = _get_coeff_cmap()
    titles = [r'Component $%s$' % (ii + 1) for ii
              in np.arange(X_comp.shape[0])]
    _draw_fields(X_comp, cmap, fontsize, titles, figsize=figsize)


def draw_correlations(X_corr, correlations=None):
    """
    Visualize spatial correlations.

    Args:
        X_corr (ND array): correlations
        correlations (list, optional): correlation labels
    """
    if correlations is None:
        n_cross = X_corr.shape[-1]
        L = range((np.sqrt(1 + 8 * n_cross) - 1).astype(int) / 2)
        correlations = zip(*list(_auto_correlations(L)))
        correlations += zip(*list(_cross_correlations(L)))
    _draw_stats(X_corr, correlations=correlations)


def draw_autocorrelations(X_auto, autocorrelations=None):
    """
    Visualize spatial autocorrelations.

    Args:
        X_auto (ND array): autocorrelations
        autocorrelations (list, optional): autocorrelation labels.
    """
    if autocorrelations is None:
        n_states = X_auto.shape[-1]
        autocorrelations = zip(*list(_auto_correlations(n_states)))
    _draw_stats(X_auto, correlations=autocorrelations)


def draw_crosscorrelations(X_cross, crosscorrelations=None):
    """
    Visualize spatial crosscorrelations.

    Args:
        X_cross (ND array): cross-correlations
        correlations (list, optional): cross-correlation labels.
    """
    if crosscorrelations is None:
        n_cross = X_cross.shape[-1]
        n_states = (np.sqrt(1 + 8 * n_cross) + 1).astype(int) // 2
        crosscorrelations = zip(*list(_cross_correlations(n_states)))
    _draw_stats(X_cross, correlations=crosscorrelations)


def _draw_stats(X_, correlations=None):
    """Visualize the spatial correlations.

    Args:
        X_: correlations
        correlations: list of tuples to select the spatial correlations
            that will be displayed.
    """
    plt.close('all')
    X_cmap = _get_coeff_cmap()
    n_plots = len(correlations)
    x_loc, x_labels = _get_ticks_params(X_.shape[0])
    y_loc, y_labels = _get_ticks_params(X_.shape[1])
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))
    if n_plots == 1:
        axs = list([axs])
    for ax, label, img in zip(axs, correlations, np.rollaxis(X_, -1)):
        ax.grid(False)
        ax.set_xticks(x_loc)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.set_yticks(y_loc)
        ax.set_yticklabels(y_labels, fontsize=12)
        im = ax.imshow(img, cmap=X_cmap, interpolation='none')
        ax.set_title(r"Correlation $l = {0}$, $l' = {1}$".format(label[0],
                                                                 label[1]),
                     fontsize=15)
        fig.subplots_adjust(right=0.8)
        divider = make_axes_locatable(ax)

        cbar_ax = divider.append_axes("right", size="10%", pad=0.05)
        cbar_ticks = _get_colorbar_ticks(img, 5)
        cbar_ticks_diff = cbar_ticks[-1] - cbar_ticks[0]
        cbar_top, cbar_grids = np.max(X_) * 0.005, 0.005
        if cbar_ticks_diff <= 1e-15:
            cbar_top = 0.
            cbar_grids = 0.5
        try:
            cbar = plt.colorbar(im, cax=cbar_ax, ticks=cbar_ticks,
                                boundaries=np.arange(cbar_ticks[0],
                                                     cbar_ticks[-1] + cbar_top,
                                                     cbar_ticks_diff *
                                                     cbar_grids))
            cbar.ax.tick_params(labelsize=12)
        except:
            cbar = plt.colorbar(im, cax=cbar_ax, boundaries=np.unique(X_))
        cbar.ax.tick_params(labelsize=12)
        fig.subplots_adjust(right=0.8)
        plt.tight_layout()
    plt.show()


def _get_ticks_params(l):
    """Get tick locations and labels for spatial correlation plots.

    Args:
        l: shape of array along the axis
    """
    segments = np.roll(np.arange(4, 7, dtype=int), 1, 0)
    m = segments[np.argmin(l % segments)]
    n = int(max((l + 1) / m, 1))
    tick_loc = list(range(0, l + n, n))
    tick_labels = list(range(int(round(- (l - 1) / 2)),
                       int(round(int((l + 1) / 2 + n))), n))
    return tick_loc, tick_labels


def _get_colorbar_ticks(X_, n_ticks):
    """
    Helper function to get colorbar color tick locations.

    Args:
        X: sspatial correlations array
           (n_samples, x,  y, local_state_correlation)
    """
    tick_range = np.linspace(np.min(X_), np.max(X_), n_ticks)
    return tick_range.astype(float)


def draw_learning_curves(estimator, X, y, ylim=None, cv=None, n_jobs=1,
                         scoring=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """Code taken from scikit-learn examples for version 0.15.

    Generate a simple plot of the test and traning learning curve.

    Args:
        estimator (class): object type that implements the "fit" and "predict"
            methods
            An object of that type which is cloned for each validation.
        title (str): Used for the title for the chart.
        X (2D array): array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y (1D array): array-like, shape (n_samples) or (n_samples,
            n_features), optional Target relative to X for classification or
            regression; None for unsupervised learning.
        ylim (tuple, optional): Defines minimum and maximum yvalues plotted.
        cv (int, optional): If an integer is passed, it is the number of folds
            (defaults to 3). Specific cross-validation objects can be passed,
            see sklearn.cross_validation module for the list of possible
            objects
        n_jobs(int, optional) : Number of jobs to run in parallel (default 1).
        train_sizes (float): Relative or absolute numbers of training examples
            that will be used to generate the learning curve. If the dtype is
            float, it is regarded as a fraction of the maximum size of the
            training set (that is determined by the selected validation
            method), i.e. it has to be within (0, 1]. Otherwise it is
            interpreted as absolute sizes of the training sets. Note that for
            classification the number of samples usually have to be big enough
            to contain at least one sample from each class. (default:
            np.linspace(0.1, 1.0, 5))

        Returns:
            A plot of the learning curves for both the training curve and the
            cross-validation curve.
    """
    plt.close('all')
    flat_shape = (X.shape[0],) + (np.prod(X.shape[1:]),)
    X_flat = X.reshape(flat_shape)
    plt.figure()
    plt.title('Learning Curves', fontsize=20)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_flat, y, cv=cv, n_jobs=n_jobs,
        train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#f46d43")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="#1a9641")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#f46d43",
             linewidth=2, label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#1a9641",
             linewidth=2, label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
