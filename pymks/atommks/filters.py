"""
@author: ashanker9@gatech.edu
"""

import torch
import numpy as np
from toolz.curried import curry, pipe


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


@curry
def conjugate(x):
    """
    For generating the conjugate
    of a complex torch tensor
    """
    y = torch.empty_like(x)
    y[..., 1] = x[..., 1] * -1
    y[..., 0] = x[... , 0]
    return y


@curry
def mult(x1, x2):
    """
    For multiplying imaginary arrays using PyTorch
    """
    y = torch.empty_like(x1)
    y[..., 0] = x1[..., 0]*x2[..., 0] - x1[..., 1]*x2[..., 1]
    y[..., 1] = x1[..., 0]*x2[..., 1] + x1[..., 1]*x2[..., 0]
    return y

to_torch = lambda x: torch.from_numpy(x).double()


@curry
def imfilter(arg1, arg2):
    """
    For convolving f_data1 with f_data2 using PyTorch
    """
    arg1 = to_torch(arg1)
    arg2 = to_torch(arg2)

    fftshift = curry(np.fft.fftshift)
    rfft = curry(torch.rfft)(signal_ndim=arg1.ndim)
    irfft = curry(torch.irfft)(signal_ndim=arg2.ndim)

    return pipe(arg1,
                rfft,
                lambda x: mult(x, conjugate(rfft(arg2))),
                irfft,
                fftshift)