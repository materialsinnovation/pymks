let
  pkgs = import (builtins.fetchTarball {
    url = https://github.com/NixOS/nixpkgs/archive/19.03.tar.gz;
    sha256 = "0q2m2qhyga9yq29yz90ywgjbn9hdahs7i8wwlq7b55rdbyiwa5dy";
  }) {};
  pypkgs = pkgs.python3Packages;
  # Sfepy is in process of being added to Nixpkgs
  sfepy = pypkgs.buildPythonPackage rec {
    name = "sfepy_${version}";
    version = "2019.2";
    src = pkgs.fetchurl {
      url="https://github.com/sfepy/sfepy/archive/release_${version}.tar.gz";
      sha256 = "17dj0wbchcfa6x27yx4d4jix4z4nk6r2640xkqcsw0mf62x5l1pj";
    };
    doCheck = false;
    buildInputs = with pypkgs; [
      numpy
      cython
      scipy
      matplotlib
      pyparsing
      tables
    ];
    catchConflicts = false;
  };
  # This nbval fix is not required for latest master branch of Nixpkgs (33b67761be99)
  nbval = (pypkgs.nbval.overridePythonAttrs ({ nativeBuildInputs = [ pypkgs.pytest ]; }));
in
  pypkgs.buildPythonPackage rec {
    pname = "pymks";
    version = "0.3.4.dev";
    nativeBuildInputs =  with pypkgs; [
      numpy
      scipy
      pytest
      matplotlib
      sympy
      cython
      jupyter
      pytestcov
      nbval
      pkgs.pkgs.git
      tkinter
      setuptools
      sfepy
      toolz
      dask
      pylint
      flake8
      pyfftw
      scikitlearn
      dask-ml
      pandas
      dask-glm
      multipledispatch
      pkgs.graphviz
      graphviz
      distributed
      black
      appdirs
      toml
      tkinter
      ipywidgets
      pip
    ];
    src=builtins.filterSource (path: type: type != "directory" || baseNameOf path != ".git") ./.;
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
