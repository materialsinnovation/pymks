let
  nixpkgs = import ./nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  pytest-cov = import ./nix/pytest-cov.nix { inherit nixpkgs pypkgs; };
  nbval = import ./nix/nbval.nix { inherit nixpkgs pypkgs; };
  scipy = import ./nix/scipy.nix { inherit nixpkgs pypkgs; };
  sfepy = import ./nix/sfepy.nix { inherit nixpkgs pypkgs; };
  dklearn = import ./nix/dklearn.nix { inherit nixpkgs pypkgs; };
  sklearn = import ./nix/sklearn.nix { inherit pypkgs; };
  dask-searchcv = import ./nix/dask-searchcv.nix { inherit pypkgs sklearn; };
  dask-ml = import ./nix/dask-ml.nix { inherit pypkgs sklearn dask-searchcv dask-glm; };
  dask-glm = import ./nix/dask-glm.nix { inherit pypkgs scipy sklearn; };
  toml = import ./nix/toml.nix { inherit pypkgs; };
  black = import ./nix/black.nix { inherit pypkgs toml; };
in
  pypkgs.buildPythonPackage rec {
    pname = "pymks";
    version = "0.3.4.dev";
    env = nixpkgs.buildEnv { name=pname; paths=buildInputs; };
    buildInputs =  [
      pypkgs.numpy
      scipy
      pypkgs.pytest
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
      pypkgs.toolz
      pypkgs.dask
      pypkgs.pylint
      pypkgs.flake8
      pypkgs.pyfftw
      dklearn
      sklearn
      dask-ml
      dask-searchcv
      pypkgs.pandas
      dask-glm
      pypkgs.multipledispatch
      nixpkgs.graphviz
      pypkgs.graphviz
      pypkgs.distributed
      black
      pypkgs.appdirs
      toml
      nixpkgs.python36Packages.tkinter
      pypkgs.ipywidgets
    ];
    src=./.;
    catchConflicts=false;
    doCheck=false;
    preShellHook = ''
      jupyter nbextension install --py widgetsnbextension --user
      jupyter nbextension enable widgetsnbextension --user --py
    '';
  }
