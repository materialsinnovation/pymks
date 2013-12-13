from mksRegressionModel import MKSRegressionModel
import numexpr as ne
import numpy as np

class FastMKSRegressionModel(MKSRegressionModel):
    def __init__(self, Nbin=10, client=None):
        super(FastMKSRegressionModel, self).__init__(Nbin=Nbin)
        self.client = client
        
    def _bin(self, X):
        """
        Bin the microstructure.

        >>> Nbin = 10
        >>> np.random.seed(4)
        >>> X = np.random.random((2, 5, 3, 2))
        >>> X_ = FastMKSRegressionModel(Nbin)._bin(X)
        >>> H = np.linspace(0, 1, Nbin)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        """
        H = np.linspace(0, 1, self.Nbin)
        dh = H[1] - H[0]
        Xtmp = X[..., None]
        tmp = ne.evaluate("1. - abs(Xtmp - H) / dh")
        return ne.evaluate("(tmp > 0) * tmp")

    def _binfft(self, X):
        r"""
        Bin the microstructure and take the Fourier transform.

        >>> Nbin = 10
        >>> np.random.seed(3)
        >>> X = np.random.random((2, 5, 3))
        >>> FX_ = FastMKSRegressionModel(Nbin)._binfft(X)
        >>> X_ = np.fft.ifftn(FX_, axes=(1, 2))
        >>> H = np.linspace(0, 1, Nbin)
        >>> Xtest = np.sum(X_ * H[None,None,None,:], axis=-1)
        >>> assert np.allclose(X, Xtest)

        """
        Xbin = self._bin(X)
        return self.fftn(Xbin, axes=self._axes(X))

    def fftn(self, a, axes):
        import pyfftw
        a_aligned = pyfftw.n_byte_align_empty(a.shape, 16, 'complex128')
        a_aligned[:] = a
        return np.fft.fftn(a_aligned, axes=axes)

    
        # if self.client:
        #     Xbin_swap = Xbin.swapaxes(0, -1)
        #     def fftn(xb):
        #         return np.fft.fftn(xb, axes=(1, 2))
        #     view = self.client[:]
        #     list_ = view.map_sync(fftn, Xbin_swap)
        #     for i, ele in enumerate(list_):
        #         list_[i] = ele.swapaxes(0, -1)[...,None]
        # else:
        #     list_ = [np.fft.fftn(Xbin[...,bin], axes=self._axes(X))[...,None] for bin in range(Xbin.shape[-1])]
        # out = np.concatenate(list_, axis=-1)
        # print out.shape

        # return out

if __name__ == '__main__':
    import fipy.tests.doctestPlus
    exec(fipy.tests.doctestPlus._getScript())
