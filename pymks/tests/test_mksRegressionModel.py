from pymks import MKSRegressionModel
import numpy as np


def test():
    Nbin = 2
    Nspace = 81
    Nsample = 400

    def filter(x):
        return np.where(x < 10,
                        np.exp(-abs(x)) * np.cos(x * np.pi),
                        np.exp(-abs(x - 20)) * np.cos((x - 20) * np.pi))

    coeff = np.linspace(1, 0, Nbin)[None,:] * filter(np.linspace(0, 20, Nspace))[:,None]
    Fcoeff = np.fft.fft(coeff, axis=0)

    np.random.seed(2)
    X = np.random.random((Nsample, Nspace))

    H = np.linspace(0, 1, Nbin)
    X_ = np.maximum(1 - abs(X[:,:,None] - H) / (H[1] - H[0]), 0)
    FX = np.fft.fft(X_, axis=1)
    Fy = np.sum(Fcoeff[None] * FX, axis=-1)
    y = np.fft.ifft(Fy, axis=1).real

    model = MKSRegressionModel(Nbin=Nbin)
    model.fit(X, y)

    assert np.allclose(np.fft.fftshift(coeff, axes=(0,)), model.coeff)

if __name__ == '__main__':
    test()
