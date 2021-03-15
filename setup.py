#!/usr/bin/env python

"""PyMKS - the materials knowledge system in Python

See the documenation for details at https://pymks.org
"""

from setuptools import setup, find_packages
import versioneer


PACKAGE_NAME = "pymks"


setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Materials Knowledge Systems in Python (PyMKS)",
    author="Daniel Wheeler",
    author_email="daniel.wheeler2@gmail.com",
    url="http://pymks.org",
    packages=find_packages(),
    package_data={"": ["tests/*.py"]},
    install_requires=[
        "pytest",
        "numpy",
        "dask",
        "Deprecated",
        "matplotlib",
        "scikit-learn",
        "pytest-cov",
        "nbval",
        "toolz",
    ],
    data_files=["setup.cfg"],
)
