#!/usr/bin/env bash

# Inspired by the Arch package for fftw
# https://projects.archlinux.org/svntogit/packages.git/tree/trunk/PKGBUILD?h=packages/fftw

# When I build fftw on Arch, it uses a later version of glibc than the anaconda
# packages use. This is problematic because it tries to link not with the
# system libm.so, but with the version provided by the "system" conda package.
# As of 2014-02-18, it looks for symbols provided by GLIBC 2.15, which the
# conda libm.so does not provide (__log_finite). So the approach is to override
# the include directories so fftw uses the math.h provided by the "system"
# conda package instead of the current Arch math.h.
CPPFLAGS+=" -I $SYS_PREFIX/include"
export CPPFLAGS=$CPPFLAGS

# Another problem with building on Arch. It uses a memcpy provided by the
# versioned symbol GLIBC_2.14, which is from string.h. There is no conda
# package that provides this string.h/libc.so, so we have to take another
# approach to force an older versioned symbol. I could do something like in
# http://www.trevorpounds.com/blog/?p=103, but there are too many source files
# to modify. So instead I build on my RHEL 6 machine, NOT on Arch.
# This is why distributing binary packages across Linux distributions is so annoying!

# This is from the Arch PKGBUILD, but it looks like ./configure figures these
# out anyway, so there isn't any need to export this CFLAGS.
#CFLAGS+=" -O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math"

# Default args to ./configure
CONFIGURE="./configure --prefix=$PREFIX --enable-shared --enable-threads --disable-fortran"

# Single precision (fftw libraries have "f" suffix)
$CONFIGURE --enable-float --enable-sse --enable-avx
make
make install

# Long double precision (fftw libraries have "l" suffix)
$CONFIGURE --enable-long-double
make
make install

# Double precision (fftw libraries have no precision suffix)
$CONFIGURE --enable-sse2 --enable-avx
make
make install

# Test suite
# This is not done in meta.yaml because it runs *after* the conda package is
# created and installed. But the fftw test suite is not included as part of the
# conda package, so there are no tests to run. So instead we run them here
# while we have access to the fftw test suite.

# two random checks
cd tests && make check-local

## 30 random checks
#cd tests && make smallcheck
#
## all checks
#cd tests && make bigcheck
