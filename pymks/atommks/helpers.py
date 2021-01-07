"""
@author: ashanker9@gatech.edu
"""
import numba
import pickle
import numpy as np
from toolz.curried import curry, pipe

try:
    import edt
    transform_edt = curry(edt.edt)(black_border=False)
except ImportError:
    import scipy.ndimage.morphology as morph
    transform_edt = morph.distance_transform_edt
    print("you can install edt for speed-up as - ")
    print("pip install edt")
    pass

try:
    import torch
    torch_rfft = curry(torch.rfft) # pylint: disable=invalid-name
    torch_irfft = curry(torch.irfft) # pylint: disable=invalid-name

    def conjugate(x):
        """
        returns the conjugate of a complex torch tensor
        """
        y = torch.empty_like(x)
        y[..., 1] = x[..., 1] * -1
        y[..., 0] = x[... , 0]
        return y

    @curry
    def mult(x1, x2):
        """
        returns the product of complext torch tensors x1 and x2
        """
        y = torch.empty_like(x1)
        y[..., 0] = x1[..., 0]*x2[..., 0] - x1[..., 1]*x2[..., 1]
        y[..., 1] = x1[..., 0]*x2[..., 1] + x1[..., 1]*x2[..., 0]
        return y

    @curry
    def imfilter(f_data1, f_data2):
        """
        For convolving f_data1 with f_data2 using PyTorch
        """
        ndim = f_data1.ndim
        f_data1 = torch.from_numpy(f_data1).double()
        f_data2 = torch.from_numpy(f_data2).double()

        rfft = torch_rfft(signal_ndim=ndim)
        irfft = torch_irfft(signal_ndim=ndim)

        return pipe(f_data1,
                    rfft,
                    lambda x: mult(x, conjugate(rfft(f_data2))),
                    irfft,
                    lambda x: np.fft.fftshift(x))
except ImportError:
    @curry
    def imfilter(f_data1, f_data2):
        """
        to convolve f_data over x_data
        """
        return pipe(f_data2,
                    lambda x: np.fft.ifftshift(x),
                    lambda x: np.fft.fftn(x),
                    lambda x: np.conj(x) * np.fft.fftn(f_data1),
                    lambda x: np.fft.ifftn(x),
                    lambda x: np.absolute(x))
    print("you can install torch for speed-up as - ")
    print("conda install pytorch-cpu torchvision-cpu -c pytorch")
    pass


def sphere(r=10):
    """
    args: radius of the sphere

    returns: A 3D cubic matric of dim (2*r+1)^1
    """
    return pipe(2*r+1,
                lambda x: np.mgrid[:x,:x,:x],
                lambda xx: (xx[0]-r)**2 + (xx[1]-r)**2+(xx[2]-r)**2,
                lambda x: (x<r*r)*1)


@curry
def padder(inp, shape, const_val=0):
    """
    args :  input matrix, new shape

    returns : matrix reshaped to given shape
    """
    ls = np.floor((shape - inp.shape) / 2).astype(int)
    hs = np.ceil((shape - inp.shape) / 2).astype(int)
    return np.pad(inp, ((ls[0], hs[0]), (ls[1], hs[1]), (ls[2], hs[2])), 'constant', constant_values=const_val)


# @curry
# def return_slice(x_data, cutoff):

#     s  = (np.asarray(x_data.shape) // 2 + 1).astype(int)
#     cutoff = (np.asarray(cutoff) // 2 + 1).astype(int)

#     if x_data.ndim == 2:
#         return x_data[(s[0] - cutoff[0]):(s[0] + cutoff[0] + 1),
#                       (s[1] - cutoff[1]):(s[1] + cutoff[1] + 1)]
#     elif x_data.ndim ==3:
#         return x_data[(s[0] - cutoff[0]):(s[0] + cutoff[0] + 1),
#                       (s[1] - cutoff[1]):(s[1] + cutoff[1] + 1),
#                       (s[2] - cutoff[2]):(s[2] + cutoff[2] + 1)]
#     else:
#         print('Incorrect Number of Dimensions!')
        
     
@curry
def return_slice(x_data, cutoff=5, s=None):
    """
    returns region of interest around the center voxel 
    upto the cutoff length
    """
    
    if not s:
        s = np.asarray(x_data.shape).astype(int) // 2
    
    if x_data.ndim == 2:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1)]
    elif x_data.ndim ==3:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1),
                      (s[2] - cutoff):(s[2] + cutoff+1)]
    else:
        print('Incorrect Number of Dimensions!')


@curry
def write2vtk(matrix, fname="zeo.vtk"):
    """
    args:
    matrix: numpy ndArray
    fname : filename
    """
    sx, sy, sz = matrix.shape
    mx = np.max(matrix)
    mi = np.min(matrix)
    lines ='# vtk DataFile Version 2.0\nVolume example\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\nASPECT_RATIO 1 1 1\nORIGIN 0 0 0\nPOINT_DATA %d\nSCALARS matlab_scalars float 1\nLOOKUP_TABLE default\n'%(sx, sy, sz, matrix.size)
    with open(fname, 'w') as f:
        f.write(lines)
        for ix in range(sz):
            v = np.ravel(matrix[:,:,ix], order="f")
            v = ["%1.5f"%x for x in np.round(100 * v / mx)]
            line = " ".join(v)
            f.write(line+"\n")

### ******* New code added to support rdf signal smoothening ********

@curry
def epanechnikov_fn(u, h):
    p = 0.75*(5**(-0.5))*(1 - (u**2)/(5*h**2))/h
    p[p<0] = 0
    return p


@curry
def epanechnikov_kernel(width, x):
    steps = x[1:] - x[:-1]
    width = width/np.sqrt(5)
    k_half = int((np.sqrt(5)*width)//steps[0] + 1)
    u = np.linspace(-k_half*steps[0],k_half*steps[0],k_half*2+1)
    kernel = epanechnikov_fn(u=u, h=width)
    shift_to_mid = -len(kernel)//2 + 1
    kernel_arr = np.zeros(len(x))
    kernel_arr[:len(kernel)] = kernel
    kernel_arr = np.roll(kernel_arr, shift_to_mid)
    return kernel_arr


@curry
def convolve_kernel(kernel_arr, sig):
    h1 = np.fft.fftn(kernel_arr)
    h2 = np.fft.fftn(sig)
    density = np.fft.ifftn(h1.conj() * h2).real
    return density

## General purpose helpers

@curry
def save_file(fname, obj):
    """
    save python object as a pickle
    """
    with open(fname, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_file(fname):
    """
    load python object from pickle file
    """
    with open(fname, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

