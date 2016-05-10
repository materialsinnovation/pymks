#!/usr/bin/env bash

$PYTHON setup.py build
$PYTHON setup.py build_ext --inplace # needed for tests below
$PYTHON setup.py install --optimize=1

# Run included test suite
# NOTE that for v0.9.2, several tests fail, but for git master (as of 2014-01-13), all the tests pass, but it takes about 10 minutes to complete
# Make sure to link against the libfftw and libm provided by conda, not the system ones
#export LD_LIBRARY_PATH="$SYS_PREFIX/lib:$LD_LIBRARY_PATH"
#$PYTHON setup.py test

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
