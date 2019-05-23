let
  nixpkgs = import ./nix/nixpkgs_version.nix;
  # Use regular blas instead of openblas (the default is
  # nixpkgs.pkgs.openblas)due to a seemingly broken dot product for
  # large matrices when using Scikit-learn.
  numpy = pypkgs.numpy.override { blas = nixpkgs.pkgs.blas; };
  pypkgs = nixpkgs.python36Packages;
  pytest-cov = import ./nix/pytest-cov.nix { inherit nixpkgs pypkgs; };
  nbval = import ./nix/nbval.nix { inherit nixpkgs pypkgs; };
  scipy = import ./nix/scipy.nix { inherit nixpkgs pypkgs; };
  sfepy = import ./nix/sfepy.nix { inherit nixpkgs pypkgs; };
  dklearn = import ./nix/dklearn.nix { inherit pypkgs; };
  dask-searchcv = import ./nix/dask-searchcv.nix { inherit pypkgs; };
  dask-ml = import ./nix/dask-ml.nix { inherit pypkgs dask-searchcv dask-glm; };
  dask-glm = import ./nix/dask-glm.nix { inherit pypkgs scipy; };
  toml = import ./nix/toml.nix { inherit pypkgs; };
  black = import ./nix/black.nix { inherit pypkgs toml; };
  distributed = import ./nix/distributed.nix { inherit pypkgs; };
in
  pypkgs.buildPythonPackage rec {
    pname = "pymks";
    version = "0.3.4.dev";
    env = nixpkgs.buildEnv { name=pname; paths=buildInputs; };
    buildInputs =  [
      numpy
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
      pypkgs.scikitlearn
      dask-ml
      dask-searchcv
      pypkgs.pandas
      dask-glm
      pypkgs.multipledispatch
      nixpkgs.graphviz
      pypkgs.graphviz
      distributed
      black
      pypkgs.appdirs
      toml
      nixpkgs.python36Packages.tkinter
      pypkgs.ipywidgets
    ];
    src=builtins.filterSource (path: type: type != "directory" || baseNameOf path != ".git") ./.;
    catchConflicts=false;
    doCheck=false;
    preShellHook = ''
      jupyter nbextension install --py widgetsnbextension --user
      jupyter nbextension enable widgetsnbextension --user --py

      export OMPI_MCA_plm_rsh_agent=/usr/bin/ssh

      SOURCE_DATE_EPOCH=$(date +%s)
      export PYTHONUSERBASE=$PWD/.local
      export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
      export PYTHONPATH=$PYTHONPATH:$USER_SITE
      export PATH=$PATH:$PYTHONUSERBASE/bin

      # To install extra packages use
      #
      # $ pip install --user <package>

    '';
  }
