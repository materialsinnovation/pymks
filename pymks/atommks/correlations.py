"""
@author: ashanker9@gatech.edu
"""
import torch
import numpy as np
from toolz.curried import pipe, curry

torch_rfft = curry(torch.rfft)
torch_irfft = curry(torch.irfft)
fftshift = curry(np.fft.fftshift)
fabs = curry(np.absolute)


def conjugate(x):
    """
    returns conjugate of the complex torch tensor x
    """
    y = torch.empty_like(x)
    y[..., 1] = x[..., 1] * -1
    y[..., 0] = x[... , 0]
    return y


@curry
def mult(x1, x2):
    """
    returns product of complex torch tensors x1 and x2
    """
    y = torch.empty_like(x1)
    y[..., 0] = x1[..., 0]*x2[..., 0] - x1[..., 1]*x2[..., 1]
    y[..., 1] = x1[..., 0]*x2[..., 1] + x1[..., 1]*x2[..., 0]
    return y


@curry
def to_torch(x, device=torch.device("cpu")):
    return pipe(x,
                lambda x: x.astype(np.float32),
                lambda x: torch.from_numpy(x).float().to(device))


@curry
def convolve(f_data, g_data=None, device=torch.device("cpu")):
    """
    Returns auto-correlation or cross-correlation of the input spatial fields
    """
    ndim = f_data.ndim

    if g_data is not None:
        g_data = to_torch(g_data, device)
        func   = lambda x: torch_rfft(signal_ndim=ndim)(g_data)
    else:
        func = lambda x: x

    return pipe(f_data,
                to_torch(device=device),
                torch_rfft(signal_ndim=ndim),
                lambda x: mult(x, conjugate(func(x))),
                torch_irfft(signal_ndim=ndim),
                lambda x: x.cpu().numpy(),
                fftshift,
                lambda x: fabs(x).astype(float))


@curry
def return_slice(x_data, cutoff):
    """
    returns region of interest around the center voxel upto the cutoff length
    """

    s = np.asarray(x_data.shape).astype(int) // 2

    if x_data.ndim == 2:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1)]
    elif x_data.ndim ==3:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1),
                      (s[2] - cutoff):(s[2] + cutoff+1)]



@curry
def compute_statistics(boundary="periodic", corrtype="auto", cutoff=None, device=torch.device("cpu"), args0=None, args1=None):
    """
    Wrapper function that returns auto or crosscorrelations for
    input fields by calling appropriate modules.
    args:
        boundary : "periodic" or "nonperiodic"
        corrtype : "auto" or "cross"
        cutoff   :  cutoff radius of interest for the 2PtStatistics field
	device   : "cpu" or "cuda"
	args0    : 2D or 3D primary field of interest
	args1    : 2D or 3D field of interest which needs to be cross-correlated with args1
    """

    ndim = args0.ndim
    size = args0.size

    if cutoff is None:
        cutoff = args0.shape[0] // 2
    cropper = return_slice(cutoff=cutoff)

    if boundary is "periodic":
        padder = lambda x: x
        if corrtype is "auto":
            y = None
        elif corrtype is "cross":
            y = args1
    elif boundary is "nonperiodic":
        padder = lambda x: np.pad(x, [(cutoff, cutoff),] * ndim, mode="constant", constant_values=0)
        if corrtype is "auto":
            y = None
        elif corrtype is "cross":
            y = padder(args1)

    return pipe(args0,
                lambda x : padder(x),
                lambda x : convolve(x, y, device=device),
                lambda x : cropper(x) / size)
