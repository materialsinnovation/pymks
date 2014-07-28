import numpy as np
import scipy as sp


def microstructure_generator(X):
    _make_filter(X)


def _make_filter(X):
    shape = np.array(X[0].shape) / 4.
    filt = np.zeros_like(X[0].shape)
    filt[:np.split(shape, len(shape))] = np.ones(shape)
    print filt


"""

    - Make filter
    - do convolution
    - assign phase


'''
    David Brough 3.1.14
    Microstructure Generating Funcion
    This function take a 2-D or 3-D matrix A, and the number of phases
    H, the microstructure form (isotropic or anisotropic, anisotropic2),
    the relative  grain size and the distribtion of the phases n, and
    returns an Eigen microstructure DMS, the the dimensions of A with H phases.

    The variable, n, can be thought of as volume fraction distribution
    factor. When n = 1 the volume fraction for all H phases is equal. As
    n increases, the volume fraction for the first phase increases and the
volume fraction for all other phases decreases.
    '''

def MS( A,H,n,r,Iso1,Iso2):

    Dim = np.array(A.shape,dtype = 'float')
    l = len(Dim)
    fDim = sp.zeros(3)

    #Checking Dimensions
    if l == 2:
        DMS = sp.rand(Dim[0],Dim[1])
        s = (np.ceil(2.0/3.0*Dim[0]),np.ceil(2.0/3.0*Dim[1]))
        f = sp.zeros(s)
        fDim = [f.shape[0],f.shape[1]]
    elif l == 3:
        DMS = sp.rand(Dim[0],Dim[1],Dim[2])
        s = (np.ceil(2*Dim[0]/3),np.ceil(2*Dim[1]/3),np.ceil(2*Dim[2]/3))
        f = sp.zeros(s) #Will be used as a filter later
        fDim = [f.shape[0],f.shape[1],f.shape[2]]
    else:
        print 'Error: The input matrix A must must be either a 2-D or 3-d'
        return

    #Confirming there is at least 2 phases
    if H < 2:
        print 'Error: The number of phases, H, must be greater than 1'
        return

    #Confirm r is sufficiently small
    if min(Dim)/3.0 < r:
        print 'r =',r
        print 'Error: The value of the parameter r must be a non-negative integer smaller than 1/3 the smallest dimenion of the input matrix, A.'
        return

    #Confirm that Iso1 and Iso2 are reasonably sized
    if Iso1 > Dim[1]:
        print 'Second dimension of microstructure =',int(Dim[1])
        print 'Iso1 =',Iso1
        print 'Error: Iso1 cannot be larger than the second dimension of the microstructure.'
        return
    elif Iso1 > r:
        print 'r =',r
        print 'Iso1 =',Iso1
        print 'Error: Iso1 cannot be larger than r.'
        return
    elif l == 3:
        if Iso2 > Dim[2]:
            print 'Third dimension of microstructure =',int(Dim[2])
            print 'Iso2 =', Iso2
            print 'Error: Iso2 cannot be larger than the third dimension of the microstructure.'
            return
        elif Iso2 > r:
            print 'r =',r
            print 'Iso2 =',Iso2
            print 'Error: Iso2 cannot be larger than r.'
            return

    P = sp.zeros(H) #Percentage of volume fractions
    for ii in range(H-1):
        P[ii+1]=(1/float(H))**(n**(float(np.sqrt(ii))))

    SumP = np.cumsum(P)
    Ptot = SumP[H-1]
    P[0]=(1-Ptot)
    Pval = np.cumsum(P)
    #Check to make sure that there are still H number of phases
    if P[H-1] < 2e-2:
        print 'Warning: The value of the parameter n you have choosen is too large for requested number of phases, H.'

    if l == 2:
        #Note - mgrid[startingValue:numberOfValues,tartingValue:numberOfValues,tartingValue:numberOfValues] Different from matlab
        X,Y = np.mgrid[-np.floor(fDim[0]/2):np.floor(fDim[0]-1),-np.floor(fDim[1]/2):np.floor(fDim[1]-1)];
        f = np.sqrt(X**2+Y**2)
        s = (Dim[0],Dim[1])
        cf = sp.zeros(s) # filter used in convolution
        cf[0:2*r,0:2*np.ceil(r/Iso1)] = \
        f[np.ceil(fDim[0]/2)-r:np.ceil(fDim[0]/2)+r,
          np.ceil(fDim[1]/2)-np.ceil(r/Iso1):np.ceil(fDim[1]/2)+np.ceil(r/Iso1)]

    elif l == 3:
        #Note - mgrid[startingValue:numberOfValues,tartingValue:numberOfValues,tartingValue:numberOfValues] Different from matlab
        X,Y,Z = np.mgrid[-np.floor(fDim[0]/2):np.floor(fDim[0]-1),-np.floor(fDim[1]/2):np.floor(fDim[1]-1),-np.floor(fDim[2]/2):np.floor(fDim[2])-1];
        f = np.sqrt(X**2+Y**2+Z**2)
        s = (Dim[0],Dim[1],Dim[2])
        cf = sp.zeros(s) # filter used in convolution
        cf[0:2*r,0:2*np.ceil(r/Iso1),0:2*np.ceil(r/Iso2)] = \
        f[np.ceil(fDim[0]/2)-r:np.ceil(fDim[0]/2)+r,
          np.ceil(fDim[1]/2)-np.ceil(r/Iso1):np.ceil(fDim[1]/2)+np.ceil(r/Iso1),\
          np.ceil(fDim[2]/2)-np.ceil(r/Iso2):np.ceil(fDim[2]/2)+np.ceil(r/Iso2)]

    DMS1 = np.fft.ifftn(np.fft.fftn(cf)*np.fft.fftn(DMS))
    DMS2 = np.fft.ifftn(np.fft.fftn(cf)*np.fft.fftn(DMS1))
    D = DMS2.min()
    E = sp.ones(Dim)*D
    DMS2 = DMS2-E
    C = DMS2.max()
    DMS3 = DMS2/C
    np.insert(Pval,0,0)

    for ii in range(H):
        DMS3[DMS3<=Pval[ii]]= ii+3
    DMS3 = DMS3-sp.ones(Dim)*2
    return np.floor(DMS3.real)

#TO Do
#r=Iso1=Iso2=0
#r=Iso1=0
"""
