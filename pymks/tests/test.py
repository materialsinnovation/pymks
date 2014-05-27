import os


from pymks import ElasticFEModel
import numpy as np

def test_ElasticFEModel():
    X = np.zeros((1, 5, 5, 2))
    X[..., 0] = 10.
    X[0, 2, 2, 0] = 1.
    X[..., 1] = 0.3
    
    model = ElasticFEModel()
    y = model.predict(X)
    assert np.allclose(y[0, 2, 2, :], (4.1987e-1, 6.5947e-2, 0), rtol=1e-4)

if __name__ == '__main__':
    test_ElasticFEModel()
