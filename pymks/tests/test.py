import os


from pymks import ElasticFEModel
import numpy as np


def test_mshfile():
    datafile = 'test.msh'
    datapath = os.path.split(__file__)[0]
    with open(os.path.join(datapath, datafile)) as f:
        test_string = f.read()

    model = ElasticFEModel()
    GEOstring = model.createGEOstring(1, 1)
    model.createMSHfile(GEOstring)
    
    with open(model.MSHfile) as f:
        msh_string = f.read()

    assert test_string == msh_string

def test_ElasticFEModel():
    X = np.zeros((1, 5, 5, 2))
    X[..., 0] = 10.
    X[..., 1] = 0.3
    
    model = ElasticFEModel()
    model.predict(X)
    
if __name__ == '__main__':
    test_ElasticFEModel()
