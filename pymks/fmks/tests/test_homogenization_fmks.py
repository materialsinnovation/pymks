"""
The MKS homogenization module test cases

"""
import numpy as np
import dask.array as da
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from pymks.fmks.correlations import FlattenTransformer, TwoPointCorrelation
from pymks.fmks.data.cahn_hilliard import solve_cahn_hilliard
from pymks.fmks.bases.legendre import LegendreTransformer


def test_classification():
    """This test basically creates Legendre microstructures in both times:
    0 and t.  Then builds homogenization classification linkages to
    classify if newly generated microstructures are at time 0 or t
    """
    reducer = PCA(n_components=3)
    linker = LogisticRegression()
    homogenization_pipeline = Pipeline(
        steps=[
            ("discretize", LegendreTransformer(n_state=3, min_=-1.0, max_=1.0)),
            (
                "Correlations",
                TwoPointCorrelation(
                    periodic_boundary=True, cutoff=10, correlations=[(1, 1), (0, 1)]
                ),
            ),
            ("flatten", FlattenTransformer()),
            ("reducer", reducer),
            ("connector", linker),
        ]
    )
    da.random.seed(3)
    x0_phase = 2 * da.random.random((50, 21, 21), chunks=(50, 21, 21)) - 1
    x1_phase = solve_cahn_hilliard(x0_phase)
    y0_class = np.zeros(x0_phase.shape[0])
    y1_class = np.ones(x1_phase.shape[0])
    x_combined = np.concatenate((x0_phase, x1_phase))
    y_combined = np.concatenate((y0_class, y1_class))
    homogenization_pipeline.fit(x_combined, y_combined)
    x0_test = 2 * da.random.random((3, 21, 21), chunks=(3, 21, 21)) - 1
    x1_test = solve_cahn_hilliard(x0_test)
    y1_test = homogenization_pipeline.predict(x1_test)
    y0_test = homogenization_pipeline.predict(x0_test)
    assert np.allclose(y0_test, [0, 0, 0])
    assert np.allclose(y1_test, [1, 1, 1])
