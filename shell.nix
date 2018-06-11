let
  nixpkgs = import <nixpkgs> {};
  # nixpkgs = import ./builds/nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  pytest-cov = import ./builds/nix/pytest-cov.nix { inherit nixpkgs; };
  nbval = import ./builds/nix/nbval.nix { inherit nixpkgs; };
  nixpkgs-old = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/c2df8a28eec869c0f6cf10811f8d3bbc65b6dfc0.tar.gz") {};

  # scikitlearn = pypkgs.scikitlearn.overridePythonAttrs (oldAttrs: {
  #   checkPhase=''
  #   '';
  # });
in
  nixpkgs.stdenv.mkDerivation rec {
    name = "pymks-dev";
    env = nixpkgs.buildEnv { name=name; paths=buildInputs; };
    buildInputs = [
      pypkgs.numpy
      nixpkgs-old.python36Packages.scipy
      pypkgs.pytest
      pypkgs.scikitlearn
      pypkgs.matplotlib
      pypkgs.sympy
      pypkgs.cython
      pypkgs.jupyter
      pypkgs.pip
      pytest-cov
      nbval
      nixpkgs.pkgs.git
    ];
    doCheck=false;
    shellHook = ''
      # python setup.py develop
      # pip install pytest-cov
    '';
  }
