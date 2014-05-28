import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np



def draw_microstructure_discretization(
        M, a=0, s=0, Nbin=6, bound=0.016, height=1.7):
    r"""
    Creates a diagram to illustrate the binning of a continues values in local state
    space.

    Args:
        Array representing a microstructure with a continuous variable.
    Returns:
        Image of the continuous local state binned discretely in the local
        state space.
    """

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


def bin(arr, Nbin):
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
        Nbin: Integer value representing the number of local states
             in the local state space of the microstructure function.
    Returns:
        Microstructure function for array `arr`.
    """
    X = np.linspace(0, 1, Nbin)
    dX = X[1] - X[0]

    return np.maximum(1 - abs(arr[:, None] - X) / dX, 0)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    r"""
    Code taken from scikit-learn examples for version 0.15.
    
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
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        
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


