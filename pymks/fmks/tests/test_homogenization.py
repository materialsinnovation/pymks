import numpy as np
import dask.array as da
from pymks.fmks.correlations import *
from pymks.datasets import make_cahn_hilliard
import numpy as np
from pymks.fmks.bases.legendre import *
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

"""
The MKS homogenization module test cases

"""
def test_classification():
    """
This test basically creates Legendre microstructures in both times: 0 and t.
Then builds homogenization classification linkages to classify if  newly
generated microstructures are at time 0 or t
    """
    reducer = PCA(n_components=3)
    linker = LogisticRegression()
    HomogenizationPipeline=Pipeline(steps=[
    ("discretize",LegendreTransformer(n_state=3, min_=-1.0, max_=1.0) ),
    ('Correlations',TwoPointcorrelation(boundary="periodic", cutoff=10, correlations=[1,1])),
    ('flatten', FlattenTransformer()),
    ('reducer',reducer),
    ('connector',linker)
    ])
    X0, X1 = make_cahn_hilliard(n_samples=50)
    y0 = np.zeros(X0.shape[0])
    y1 = np.ones(X1.shape[0])
    X = np.concatenate((X0, X1))
    y = np.concatenate((y0, y1))
    HomogenizationPipeline.fit(X,y)
    X0_test, X1_test = make_cahn_hilliard(n_samples=3)
    y1_test = HomogenizationPipeline.predict(X1_test)
    y0_test = HomogenizationPipeline.predict(X0_test)
    assert np.allclose(y0_test, [0, 0, 0])
    assert np.allclose(y1_test, [1, 1, 1])
