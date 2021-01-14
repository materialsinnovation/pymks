#
# $ nix-shell --pure --arg withBoost false --argstr tag 20.09
#

{
  tag ? "20.03-beta",
  withBoost ? true,
  withSfepy ? true
}:

let
  pkgs = import (builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/${tag}.tar.gz") {};
  pypkgs = pkgs.python3Packages;
  sfepy_ = pypkgs.sfepy.overridePythonAttrs (old: rec {
    name = "sfepy_${version}";
    version = "2019.4";
    src = builtins.fetchurl {
      url="https://github.com/sfepy/sfepy/archive/release_${version}.tar.gz";
      sha256 = "1l9vgcw09l6bwhgfzlbn68fzpvns25r6nkd1pcp7hz5165hs6zzn";
    };
    postPatch = ''
    # broken test
    rm tests/test_homogenization_perfusion.py
    rm tests/test_splinebox.py

    # slow tests
    rm tests/test_input_*.py
    rm tests/test_elasticity_small_strain.py
    rm tests/test_term_call_modes.py
    rm tests/test_refine_hanging.py
    rm tests/test_hyperelastic_tlul.py
    rm tests/test_poly_spaces.py
    rm tests/test_linear_solvers.py
    rm tests/test_quadratures.py
    '';
  });
  boost = if withBoost then pkgs.boost else null;
  sfepy = if withSfepy then sfepy_ else null;
in
  pypkgs.buildPythonPackage rec {
    pname = "pymks";
    version = "0.3.4.dev";
    buildInputs = [
       boost
    ];
    nativeBuildInputs =  with pypkgs; [
      sfepy
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
      zarr
      boost
    ];
    src = builtins.filterSource (path: type: type != "directory" || baseNameOf path != ".git") ./.;
    doCheck = false;
    PYMKS_USE_BOOST = withBoost;
    preShellHook = ''

      export OMPI_MCA_plm_rsh_agent=/usr/bin/ssh

      SOURCE_DATE_EPOCH=$(date +%s)
      export PYTHONUSERBASE=$PWD/.local
      export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
      export PYTHONPATH=$PYTHONPATH:$USER_SITE
      export PATH=$PATH:$PYTHONUSERBASE/bin

      jupyter nbextension install --py widgetsnbextension --user > /dev/null 2>&1
      jupyter nbextension enable widgetsnbextension --user --py > /dev/null 2>&1
      pip install jupyter_contrib_nbextensions --user > /dev/null 2>&1
      jupyter contrib nbextension install --user > /dev/null 2>&1
      jupyter nbextension enable spellchecker/main > /dev/null 2>&1

      pip install --user Deprecated

      # To install extra packages use
      #
      # $ pip install --user <package>

    '';
  }
