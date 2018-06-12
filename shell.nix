let
  nixpkgs = import ./nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  pytest-cov = import ./nix/pytest-cov.nix { inherit nixpkgs; inherit pypkgs; };
  nbval = import ./nix/nbval.nix { inherit nixpkgs; inherit pypkgs; };
  scipy = import ./nix/scipy.nix { inherit nixpkgs; inherit pypkgs; };
  sfepy = import ./nix/sfepy.nix { inherit nixpkgs; inherit pypkgs; };
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
      sfepy
      pypkgs.pyfftw
    ];
    src=./.;
    shellHook = ''
      python ${src}/setup.py develop --prefix=$HOME/.local
      export VERSION=`python -V | sed -r 's/.*\s([0-9]\.[0-9])\..*/\1/g'`
      export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python$VERSION/site-packages
    '';
  }
