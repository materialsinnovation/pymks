"""Functions for generating data
"""
from ..func import curry


@curry
def solve_fe(x_data, elastic_modulus, poissons_ratio, macro_strain=1.0, delta_x=1.0):
    """Solve the elasticity problem

    Args:
      x_data: microstructure with shape (n_samples, n_x, ...)
      elastic_modulus: the elastic modulus in each phase
      poissons_ration: the poissons ratio for each phase
      macro_strain: the macro strain
      delta_x: the grid spacing

    Returns:
      a dictionary of strain, displacement and stress with stress and
      strain of shape (n_samples, n_x, ..., 3) and displacement shape
      of (n_samples, n_x + 1, ..., 2)

    """
    from .elastic_fe import _solve_fe  # pylint: disable=import-outside-toplevel

    return _solve_fe(
        x_data,
        elastic_modulus,
        poissons_ratio,
        macro_strain=macro_strain,
        delta_x=delta_x,
    )
