# distutils: language=c++
# distutils: include_dirs = ../src/
# distutils: sources = src/graph_constructors.cpp

cimport graspi

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair

from libcpp cimport bool

cimport numpy as np
ctypedef np.npy_intp INTP
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef extern from "../src/graspi_descriptors.hpp" namespace "graspi" :
    ctypedef pair[float,string] desc_t;

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef extern from "../src/graspiAPI.hpp" namespace "graspi" :
    vector[desc_t] compute_descriptors_only(int* , int nx, int ny, int nz, float pixelS, bool if_per);

def compute_descriptors(np.ndarray[np.int32_t, ndim=1] colors, nx, ny, nz, pixelS,if_per):
    return compute_descriptors_only(<int*>colors.data,nx,ny,nz,pixelS,if_per)




