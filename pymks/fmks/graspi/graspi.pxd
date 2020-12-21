# distutils: language=c++

cdef extern from "./graspiAPI.hpp":
    float compute_descriptors_only();
