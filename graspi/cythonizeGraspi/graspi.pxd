# distutils: language=c++
# distutils: include_dirs = ../src/
# distutils: sources = ../src/graph_constructors.cpp

cdef extern from "../src/graspiAPI.hpp":
    float compute_descriptors_only();


