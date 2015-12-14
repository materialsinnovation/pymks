import scipy.io as sio
import os

__all__ = ['imposter',
           'make_elastic_stress_random',
           'make_elastic_FE_strain_delta',
           'make_elastic_FE_strain_random']


def imposter():
    print 'this is a fake thing'
    return 0


def get_data(data_key):
    path = os.path.split(__file__)[0] + '/'
    _keys = ['X_' + data_key, 'y_' + data_key]
    fake_data = sio.loadmat(path + 'mock_data.mat')
    return [fake_data[k] for k in _keys]


def make_elastic_FE_strain_delta(elastic_modulus=(100, 150),
                                 poissons_ratio=(0.3, 0.3),
                                 size=(21, 21), macro_strain=0.01):
    _key = 'strain_delta'
    return get_data(_key)


def make_elastic_FE_strain_random(n_samples=1, elastic_modulus=(100, 150),
                                  poissons_ratio=(0.3, 0.3), size=(21, 21),
                                  macro_strain=0.01):
    _key = 'strain_random'
    return get_data(_key)


def make_elastic_stress_random(n_samples=[10, 10], elastic_modulus=(100, 150),
                               poissons_ratio=(0.3, 0.3), size=(21, 21),
                               macro_strain=0.01, grain_size=[(3, 3), (9, 9)],
                               seed=10):
    if poissons_ratio == (1, 1):
        _key = 'stress_1'
    elif grain_size == (1, 1):
        _key = 'stress_2'
    else:
        _key = 'stress_3'

    return get_data(_key)


if __name__ == "__main__":
    path = os.path.split(__file__)[0] + '/'
    print path
    fake_data = sio.loadmat(path + 'mock_data.mat')
    print fake_data.keys()
    print fake_data['X_stress_3'].shape

    X, y = make_elastic_stress_random()
    print X
    print y
