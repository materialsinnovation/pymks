"""Test the localization model.
"""

from sklearn.pipeline import make_pipeline
import numpy as np
import dask.array as da
from pymks.fmks.bases.primitive import discretize, redundancy
from pymks.fmks.localization import fit
from pymks.fmks.bases.primitive import PrimitiveTransformer
from pymks.fmks.localization import LocalizationRegressor
from pymks.fmks.data.delta import generate_delta
from pymks.fmks.data.elastic_fe import solve_fe


def _get_x():
    return da.from_array(np.linspace(0, 1, 4).reshape((1, 2, 2)), chunks=(1, 2, 2))


def test():
    """Very simple example.
    """
    assert np.allclose(
        fit(
            _get_x(),
            _get_x().swapaxes(1, 2),
            discretize(n_state=2),
            redundancy_func=redundancy,
        ),
        [[[0.5, 0.5], [-2, 0]], [[-0.5, 0], [-1, 0]]],
    )


def test_setting_kernel():
    """Test resetting the coeffs after coeff resize.
    """

    x_data = generate_delta(n_phases=2, shape=(21, 21)).persist()

    y_data = solve_fe(
        x_data, elastic_modulus=(100, 130), poissons_ratio=(0.3, 0.3), macro_strain=0.01
    )["strain"][..., 0].persist()

    model = make_pipeline(PrimitiveTransformer(n_state=2), LocalizationRegressor())

    shape = (30, 30)
    fcoeff = model.fit(x_data, y_data).steps[1][1].coeff
    assert np.allclose(model.steps[1][1].coeff_resize(shape).coeff.shape[:-1], shape)
    model.steps[1][1].coeff = fcoeff
    assert np.allclose(model.predict(x_data), y_data, atol=1e-4)
