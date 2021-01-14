#!/usr/bin/env python

"""PyMKS - the materials knowledge system in Python

See the documenation for details at https://pymks.org
"""

import configparser
import pathlib
import warnings
import os
import subprocess
from distutils.util import strtobool

from setuptools import setup, find_packages, Extension
from setuptools.config import read_configuration


def make_version(package_name):
    """Generates a version number using `git describe`.

    Returns:
      version number of the form "3.1.1.dev127+g413ed61".
    """

    def _minimal_ext_cmd(cmd):
        """Run a command in a subprocess.

        Args:
          cmd: list of the command

        Returns:
          output from the command
        """
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            value = os.environ.get(k)
            if value is not None:
                env[k] = value
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    version = "unknown"

    if os.path.exists(".git"):
        try:
            out = _minimal_ext_cmd(["git", "describe", "--tags", "--match", "v*"])
            # ticket:475 - fix for bytecode received in Py3k
            # http://jeetworks.org/node/67
            outdecode = out.decode("utf-8")
            # convert git long-form version string, e.g.,
            # "version-3_1_1-127-g413ed61", into PEP 440 version,
            # e.g., "3.1.1.dev127+g413ed61"
            version = outdecode.strip().split("-")
            if len(version) > 1:
                version, dev, sha = version
                version = "%s.dev%s+%s" % (version[1:], dev, sha)
            else:
                version = version[0][1:]
        except OSError:
            warnings.warn("Could not run ``git describe``")
    elif os.path.exists("pymks.egg-info"):
        # pylint: disable=import-outside-toplevel
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            version = get_distribution(
                package_name
            ).version  # pylint: disable=no-member
        except DistributionNotFound:  # pragma: no cover
            version = "unknown, try running `python setup.py egg_info`"

    return version


def get_setupcfg():
    """Get the absolute path for setup.cfg
    """
    return pathlib.Path(__file__).parent.absolute() / "setup.cfg"


def get_configuration():
    """Get contents of setup.cfg as a dict
    """

    return read_configuration(get_setupcfg())


def read_all_config(option):
    """Read all of the options in setup.cfg not just those used in setup.
    """
    parser = configparser.ConfigParser()
    parser.read(get_setupcfg())
    if parser.has_option("pymks", option):
        return parser.getboolean("pymks", option)
    return False


def env_var(var):
    """Determine the value of an enviroment variable

    Args:
      var: variable

    Returns:
      (defined, value): `defined` is bool depending on whether
        var is defined, `value` is the bool value it's set
        to (positive if undetermined)
    """
    defined = var in os.environ
    if defined:
        env_string = os.environ[var]
        try:
            value = strtobool(env_string)
        except ValueError:
            value = True
    else:
        value = False
    return defined, value


def get_name():
    """Single location for name of package
    """
    return get_configuration()["metadata"]["name"]


def build_graspi():
    """Decide whether to build Graspi
    """
    env_defined, env_value = env_var("PYMKS_USE_BOOST")
    if env_defined:
        return env_value
    return read_all_config("use-boost")


def graspi_path():
    """Find the path to graspi
    """
    return list(filter(lambda x: "graspi" in x, find_packages()))[0].replace(".", "/")


def graspi_extension():
    """Configure the graspi extension

    """
    import numpy  # pylint: disable=import-outside-toplevel

    return Extension(
        name=graspi_path().replace("/", ".") + ".graspi",
        sources=[
            os.path.join(graspi_path(), "graspi.pyx"),
            os.path.join(graspi_path(), "graph_constructors.cpp"),
        ],
        include_dirs=[numpy.get_include(), graspi_path(), "."],
        extra_compile_args=["-std=c++11"],
        language="c++",
        optional=True,
    )


def get_extensions():
    """Get all extensions, return empty dict if no extension modules
    activated

    """

    def cythonize(*args, **kwargs):
        """Only import cython if actually using cython to build.
        """
        from Cython.Build import (  # pylint: disable=import-outside-toplevel
            cythonize as cythonize_,
        )

        return cythonize_(*args, **kwargs)

    return (
        dict(
            ext_modules=cythonize(
                [graspi_extension()],
                compiler_directives={"language_level": "3"},
                include_path=[graspi_path()],
            )
        )
        if build_graspi()
        else dict()
    )


def setup_args():
    """Get the setup arguments not configured in setup.cfg
    """
    return dict(
        version=make_version(get_name()),
        packages=find_packages(),
        package_data={"": ["tests/*.py"]},
        data_files=["setup.cfg"],
        **get_extensions()
    )


setup(**setup_args())
