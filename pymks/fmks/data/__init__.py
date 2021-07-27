"""Functions for generating data
"""
from ..func import curry


@curry
def solve_fe(x_data, elastic_modulus, poissons_ratio, macro_strain=1.0, delta_x=1.0):
    r"""Solve the elasticity problem

    Use `Sfepy <https://sfepy.org/doc-devel/index.html>`_ to solve a
    linear strain problem in 2D with a varying microstructure on a
    rectangular grid. The rectangle (cube) is held at the negative
    edge (plane) and displaced by 1 on the positive x edge
    (plane). Periodic boundary conditions are applied to the other
    boundaries.

    The boundary conditions on the rectangle (or cube) are given by

    .. math::

       u(L, y) = L \left(1 + \bar{\varepsilon}_{xx}\right)

    .. math::

       u(0, L) = u(0, 0) = 0

    .. math::

       u(x, 0) = u(x, L)

    where :math:`\bar{\varepsilon}_{xx}` is the ``macro_strain``,
    :math:`u` is the displacement in the :math:`x` direction, and
    :math:`L` is the length of the domain. More details about these
    boundary conditions can be found in `Landi et al
    <http://dx.doi.org/10.1016/j.actamat.2010.01.007>`_.

    See the `elasticity notebook for a full set of equations
    <http://pymks.org/en/stable/rst/notebooks/elasticity.html#Elastostatics-Equations>`_.

    ``x_data`` should have integer values that represent the phase of
    the material. The integer values should correspond to the indices
    for the ``elastic_modulus`` and ``poisson_ratio`` sequences and,
    therefore, ``elastic_modulus`` and ``poisson_ratio`` need to be of
    the same length.

    Args:
      x_data: microstructures with shape, ``(n_samples, n_x, ...)``
      elastic_modulus: the elastic modulus in each phase, ``(e0, e1, ...)``
      poissons_ratio: the poissons ratio for each phase, ``(p0, p1, ...)``
      macro_strain: the macro strain, :math:`\bar{\varepsilon}_{xx}`
      delta_x: the grid spacing

    Returns:
      a dictionary of strain, displacement and stress with stress and
      strain of shape ``(n_samples, n_x, ..., 3)`` and displacement shape
      of ``(n_samples, n_x + 1, ..., 2)``

    >>> import numpy as np
    >>> x_data = np.zeros((1, 11, 11), dtype=int)
    >>> x_data[0, :, 1] = 0

    ``x_data`` has values of 0 and 1 and so ``elastic_modulus`` and
    ``poisson_ratio`` must each have 2 entries for phase 0 and phase
    1.

    >>> strain = solve_fe(
    ...     x_data,
    ...     elastic_modulus=(1.0, 10.0),
    ...     poissons_ratio=(0., 0.),
    ...     macro_strain=1.,
    ...     delta_x=1.
    ... )['strain']

    >>> from pymks import plot_microstructures
    >>> fig = plot_microstructures(strain[0, ..., 0], titles=r'$\varepsilon_{xx}$')
    >>> fig.show()  #doctest: +SKIP

    .. image:: strain.png
       :width: 300

    """
    from .elastic_fe import _solve_fe  # pylint: disable=import-outside-toplevel

    return _solve_fe(
        x_data,
        elastic_modulus,
        poissons_ratio,
        macro_strain=macro_strain,
        delta_x=delta_x,
    )
