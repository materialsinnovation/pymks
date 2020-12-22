"""Top level functions for the Graspi graph descriptors for materials
science.

"""
# flake8: noqa
import pandas
import numpy as np
import pytest
from toolz.curried import keymap

from ..func import fmap, curry, pipe


def graph_descriptors(data, delta_x=1.0, periodic_boundary=True):
    # pylint: disable=line-too-long
    """Compute graph descriptors for multiple samples

    Args:
      data: array of phases (n_samples, n_x, n_y), values must be 0 or 1
      delta_x: pixel size
      periodic_boundary: whether the boundaries are periodic

    Returns:
      A Pandas data frame with samples along rows and descriptors
      along columns

    Compute graph descriptors for multiple samples using the GraSPI
    sub-package. See the `installation instructions
    <rst/README.html#graspi>`_ to install PyMKS with
    GraSPI enabled.

    GraSPI is focused on characterizing photovoltaic devices and so
    the descriptors must be understood in this context. Future
    releases will have more generic descriptors. See `Wodo et
    al. <https://arxiv.org/pdf/1106.3536.pdf>`_ for more details.
    Note that the current implementation only works for two phase
    data.

    This function returns a Pandas Dataframe with the descriptors as
    columns and samples in rows. In the context of a photovoltaic
    device the top of the domain (y-direction) represents an anode and
    the bottom of the domain represents a cathode. Phase 0 represents
    donor materials while phase 1 represents acceptor material. Many
    of these descriptors characterizes the morphology in terms of hole
    electron pair generation and transport leading to device charge
    extraction.

    The column descriptors are as follows.

    ========================= ===========
    Column Name               Description
    ========================= ===========
    n_vertices                The number of vertices in the constructed graph. Should be equal to the number of pixels.
    n_edges                   The number of edges in the constructed graph.
    n_phase{i}                The number of vertices for phase {i}.
    n_phase{i}_connect        The number of connected components for phase {i}.
    n_phase{i}_connect_top    The number of connected components for phase {i} with the top of the domain in y-direction.
    n_phase{i}_connect_bottom The number of connected components for phase {i} with the top of the domain in y-direction.
    weighted_frac_phase{i}    Weighted fraction of phase {i} vertices.
    frac_phase{i}             Fraction of phase {i} vertices.
    w_frac_phase{i}_{j}_dist  Weighted fraction of phase {i} vertices within j nodes from an interface.
    frac_phase{i}_{j}_dist    Fraction of phase {i} vertices within {j} nodes from an interface.
    frac_useful               Fraction of useful vertices connected the top or bottom of the domain.
    inter_frac_bottom_top     Fraction of interface with complementary paths to bottom or top of the domain.
    frac_phase{i}_top         Fraction of phase {i} interface vertices with path to top.
    frac_phase{i}_bottom      Fraction of phase {i} interface vertices with path to bottom.
    n_inter_paths             Number of interface edges with complementary paths.
    n_phase{i}_inter_top      Number of phase {i} interface vertices with path to top
    n_phase{i}_inter_bottom   Number of phase {i} interface vertices with path to bottom
    frac_phase{i}_rising      Fraction of phase {i} with rising paths
    ========================= ===========


    Example, with 3 x (3, 3) arrays

    Read in the expected data.

    >>> from io import StringIO
    >>> expected = pandas.read_csv(StringIO('''
    ... n_vertices,n_edges,n_phase0,n_phase1,n_phase0_connect,n_phase1_connect,n_phase0_connect_top,n_phase1_connect_bottom,weighted_frac_phase0,frac_phase0,weighted_frac_phase0_10_dist,fraction_phase0_10_dist,inter_frac_bottom_and_top,frac_phase0_top,frac_phase1_bottom,n_inter_paths,n_phase0_inter_top,n_phase1_inter_bottom,frac_phase0_rising,frac_phase1_rising
    ... 9,6,3,6,1,1,0,1,0.3245655298233032,0.3333333432674408,0.9624541997909546,1.0,0.0,0.0,1.0,0,0,6,0.0,0.8333333134651184
    ... 9,6,3,6,1,2,0,1,0.32673290371894836,0.3333333432674408,0.9624541997909546,1.0,0.0,0.0,0.5,0,0,3,0.0,1.0
    ... 9,6,6,3,1,1,1,1,0.6534875631332397,0.6666666865348816,0.9624541997909546,1.0,1.0,1.0,1.0,6,6,3,1.0,1.0
    ... '''))

    Construct the 3 samples each with 3x3 voxels

    >>> data = np.array([[[0, 1, 0],
    ...                   [0, 1, 1],
    ...                   [1, 1, 1]],
    ...                  [[1, 1, 1],
    ...                   [0, 0, 0],
    ...                   [1, 1, 1]],
    ...                  [[0, 1, 0],
    ...                   [0, 1, 0],
    ...                   [0, 1, 0]]])
    >>> actual = graph_descriptors(data)

    ``graph_descriptors`` returns a data frame.

    >>> actual
       n_vertices  n_edges  ...  frac_phase0_rising  frac_phase1_rising
    0           9        6  ...                 0.0            0.833333
    1           9        6  ...                 0.0            1.000000
    2           9        6  ...                 1.0            1.000000
    <BLANKLINE>
    [3 rows x 20 columns]

    Check that the actual values are equal to the expected values.

    >>> assert np.allclose(actual, expected)

    On examining the data for this simple test case there are a few
    obvious checks. Each sample has 9 vertices since there are 9
    pixels in each sample.

    >>> actual.n_vertices
    0    9
    1    9
    2    9
    Name: n_vertices, dtype: int64

    Notice that the first and third sample have two phase 1 regions
    connected to either the top or bottom of the domain while the
    second sample has only 1 region.

    >>> actual.n_phase1_connect
    0    1
    1    2
    2    1
    Name: n_phase1_connect, dtype: int64

    All paths are blocked for the first and second samples from
    reaching the top from the bottom surface. The third sample has 6
    interface edges that connect the top and bottom.

    >>> actual.n_inter_paths
    0    0
    1    0
    2    6
    Name: n_inter_paths, dtype: int64

    """
    # pylint: enable=line-too-long
    columns = keymap(
        lambda x: x.encode("UTF-8"),
        dict(
            STAT_n="n_vertices",
            STAT_e="n_edges",
            STAT_n_D="n_phase0",
            STAT_n_A="n_phase1",
            STAT_CC_D="n_phase0_connect",
            STAT_CC_A="n_phase1_connect",
            STAT_CC_D_An="n_phase0_connect_top",
            STAT_CC_A_Ca="n_phase1_connect_bottom",
            ABS_wf_D="weighted_frac_phase0",
            ABS_f_D="frac_phase0",
            DISS_wf10_D="weighted_frac_phase0_10_dist",
            DISS_f10_D="fraction_phase0_10_dist",
            DISS_f2_D="fraction_phase0_2_dist",
            CT_f_conn_D="frac_useful",
            CT_f_e_conn="inter_frac_bottom_and_top",
            CT_f_conn_D_An="frac_phase0_top",
            CT_f_conn_A_Ca="frac_phase1_bottom",
            CT_e_conn="n_inter_paths",
            CT_e_D_An="n_phase0_inter_top",
            CT_e_A_Ca="n_phase1_inter_bottom",
            CT_f_D_tort1="frac_phase0_rising",
            CT_f_A_tort1="frac_phase1_rising",
        ),
    )
    return pipe(
        data,
        fmap(
            graph_descriptors_sample(
                delta_x=delta_x, periodic_boundary=periodic_boundary
            )
        ),
        list,
        pandas.DataFrame,
        lambda x: x.rename(columns=columns),
        lambda x: x.apply(
            lambda x: np.rint(x).astype(int) if x.name[:2] == "n_" else x
        ),
    )


@curry
def graph_descriptors_sample(data, delta_x=1.0, periodic_boundary=True):
    """Calculate graspi graph descriptors for a single array
    """
    graspi = pytest.importorskip("pymks.fmks.graspi.graspi")
    compute = lambda x: graspi.compute_descriptors(
        x,
        *(data.shape + (3 - len(data.shape)) * (1,)),
        pixelS=delta_x,
        if_per=periodic_boundary
    )
    return pipe(
        data,
        lambda x: x.astype(np.int32).flatten(),
        compute,
        fmap(lambda x: x[::-1]),
        dict,
    )
