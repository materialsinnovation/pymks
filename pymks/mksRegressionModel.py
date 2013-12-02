import numpy as np
from sklearn.linear_model import LinearRegression

class MKSRegressionModel(LinearRegression):
    r"""
    The `MKSRegressionModel` fits data using the Materials Knowledge
    System in Fourier Space. There main assumption is that the
    microstructure (`X`) must vary only between 0 and 1. Example,

    >>> Nbin = 51
    >>> Nspace = 41
    >>> Nsample = 100

    >>> coeffs = np.linspace(0, 1, Nbin)[None, :] * filter(np.linspace(-10, 10, Nspace))[:, None]

    >>> def filter(x):
    ...    return np.exp(-abs(x)) * np.cos(x * np.pi)

    >>> Fcoeffs = np.fft.fft(coeffs, axis=0)

    >>> np.random.seed(2)
    >>> microstructures = np.random.random((Nsample, Nspace))
    >>> def bin(ms, Nbins):
    ...    H = np.linspace(0, 1, Nbins)
    ...    dh = (H[1] - H[0])
    ...    return np.maximum(1 - abs(ms[:, :, np.newaxis] - H) / dh, 0)
    >>> binnedMicrostructures = bin(microstructures, Nbin)[:,:,:]

    >>> Fstructures = np.fft.fft(binnedMicrostructures, axis=1)

    >>> Fresponse = np.sum(Fcoeffs[np.newaxis] * Fstructures, axis=-1)
    
    >>> response = np.fft.ifft(Fresponse, axis=0)
    >>> Nsample, Nspace, Nbin = 6, 4, 3 
    >>> np.random.seed(0)
    >>> X = np.array([np.random.random((Nspace, Nspace)) for i in range(Nsample)])
    y = np.array([fipy_response(xi, dt=dt) for xi in x])
    return x, y
    """
    
    def __init__(self, Nbin=10):
        self.Nbin = Nbin
        
    def _bin(self, Xi):
        H = np.linspace(0, 1, self.Nbin)
        dh = H[1] - H[0]
        return np.maximum(1 - abs(Xi[:,:,np.newaxis] - H) / dh, 0)
        
    def _binfft(self, X):
        Xbin = np.array([self._bin(Xi) for Xi in X])
        return np.fft.fft2(Xbin, axes=(1, 2))
        
    def fit(self, X, y):
        if len(X.shape) == 2:
            X = X[:,:,None]
        if len(y.shape) == 2:
            y = y[:,:,None]
        Nsample, Xspace, Yspace = X.shape
        assert y.shape == (Nsample, Xspace, Yspace)
        FX = self._binfft(X)
        self.FX = FX
        Fy = np.fft.fft2(y, axes=(1, 2))
        self.Fy = Fy
        self.coeff = np.zeros((Xspace, Yspace, self.Nbin), dtype=np.complex)
        for i in range(Xspace):
            for j in range(Yspace):
                self.coeff[i,j,:] = np.linalg.lstsq(FX[:,i,j,:], Fy[:,i,j])[0]
                
    def predict(self, X):
        FX = self._binfft(X)
        Fy = np.sum(FX * self.coeff[None,...], axis=-1)
        return np.fft.ifft2(Fy, axes=(1, 2)).real
