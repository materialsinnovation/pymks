#!/usr/bin/env python

"""PyMKS - the materials knowledge system in Python

See the documenation for details at https://pymks.org
"""

import pathlib

from setuptools.config import read_configuration
from setuptools import setup, find_packages
import versioneer


def get_setupcfg():
    """Get the absolute path for setup.cfg
    """
    return pathlib.Path(__file__).parent.absolute() / "setup.cfg"


def get_configuration():
    """Get contents of setup.cfg as a dict
    """

    return read_configuration(get_setupcfg())


def get_name():
    """Single location for name of package
    """
    return get_configuration()["metadata"]["name"]


def setup_args():
    """Get the setup arguments not configured in setup.cfg
    """
    return dict(
        packages=find_packages(),
        package_data={"": ["tests/*.py"]},
        data_files=["setup.cfg"],
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )


setup(**setup_args())
