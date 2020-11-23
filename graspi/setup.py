from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

sourcefiles = ["cythonizeGraspi/graspi.pyx", "src/graph_constructors.cpp"]

extensions = [
    Extension(
        "graspi",
        sourcefiles,
        include_dirs=[numpy.get_include(), "src"],
        extra_compile_args=["-std=c++11"],
        language="c++",
    ),
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
