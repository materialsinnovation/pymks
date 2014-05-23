import os
import difflib


from pymks import ElasticFEModel


def test_mshfile():
    datafile = 'test.msh'
    datapath = os.path.split(__file__)[0]
    with open(os.path.join(datapath, datafile)) as f:
        test_string = f.read()

    model = ElasticFEModel(1)
    
    with open(model.msh_file) as f:
        msh_string = f.read()

    assert test_string == msh_string

if __name__ == '__main__':
    test_mshfile()
