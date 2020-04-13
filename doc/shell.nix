let
  nixpkgs = import ../nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  pymks = import ../default.nix;
in
  nixpkgs.stdenv.mkDerivation rec {
    name = "pymks-doc-env";
    buildInputs = [
      pypkgs.virtualenv
      pypkgs.pip
      pypkgs.recommonmark
      pypkgs.sphinx
      pypkgs.m2r
      nixpkgs.pkgs.pandoc
      pymks
    ] ++ pymks.buildInputs;
    src=null;
    shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    \rm -rf $HOME/.local
    mkdir -p $HOME/.local
    pip install --user sphinx_bootstrap_theme==0.6.5
    pip install --user sphinxcontrib-napoleon==0.6.1
    pip install --user nbsphinx==0.3.3
    export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH
    export PATH=$PATH:$HOME/.local/bin
    '';
  }
