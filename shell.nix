let
  # nixpkgs = import <nixpkgs> {};
  nixpkgs = import ./nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  pytest-cov = import ./nix/pytest-cov.nix { inherit nixpkgs; inherit pypkgs; };
  nbval = import ./nix/nbval.nix { inherit nixpkgs; inherit pypkgs; };
  scipy = import ./nix/scipy.nix { inherit nixpkgs; inherit pypkgs; };
  # nixpkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/c2df8a28eec869c0f6cf10811
  # f8d3bbc65b6dfc0.tar.gz") {};

  # scikitlearn = pypkgs.scikitlearn.overridePythonAttrs (oldAttrs: {checkPhase="";});
  python = pypkgs.python;
  scikitlearn = pypkgs.scikitlearn.overridePythonAttrs (oldAttrs: {checkPhase=''
    HOME=$TMPDIR OMP_NUM_THREADS=1 nosetests --doctest-options=+SKIP $out/${python.sitePackages}/sklearn/
  '';});
in
  nixpkgs.stdenv.mkDerivation rec {
    name = "pymks-dev";
    env = nixpkgs.buildEnv { name=name; paths=buildInputs; };
    buildInputs = [
      pypkgs.numpy
      scipy
      pypkgs.pytest
      scikitlearn
      pypkgs.matplotlib
      pypkgs.sympy
      pypkgs.cython
      pypkgs.jupyter
      pytest-cov
      nbval
      nixpkgs.pkgs.git
      pypkgs.tkinter
      pypkgs.setuptools
    ];
    src=./.;
    shellHook = ''
      python ${src}/setup.py develop --prefix=$HOME/.local
      export VERSION=`python -V | sed -r 's/.*\s([0-9]\.[0-9])\..*/\1/g'`
      export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python$VERSION/site-packages
    '';
  }
