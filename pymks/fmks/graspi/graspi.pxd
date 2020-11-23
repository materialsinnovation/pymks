# distutils: language=c++
# distutils: include_dirs = pymks/fmks/graspi
# distutils: sources = pymks/fmks/graspi/graph_constructors.cpp

cdef extern from "./graspiAPI.hpp":
    float compute_descriptors_only();
