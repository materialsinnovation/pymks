# { pkgs ? (import (builtins.fetchTarball {
#     url = "https://github.com/NixOS/nixpkgs/archive/19.09.tar.gz";
#     sha256 = "0mhqhq21y5vrr1f30qd2bvydv4bbbslvyzclhw0kdxmkgg3z4c92";
#   }) {}) }:
{ pkgs ? (import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/5a0e91e78f43484a46303f60dad4c411cdc6c7d4.tar.gz";
    sha256 = "11cak4532852cbrzws2fx7jdwxkhc1jcvvagy4sprm0zlnqsvha6";
  }) {}) }:

let
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
in
  pypkgs.buildPythonPackage rec {
    pname = "pymks";
    version = "0.3.4.dev";
    nativeBuildInputs =  with pypkgs; [
      nbval
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
      pkgs.openssh
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
