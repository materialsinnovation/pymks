let
  nixpkgs = import ../nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  default = import ../default.nix;
  my_pkgs = default.buildInputs;
in
  nixpkgs.stdenv.mkDerivation rec {
    name = "pymks-doc-env";
    buildInputs = [
      pypkgs.sphinx
      pypkgs.virtualenv
      pypkgs.pip
    ] ++ my_pkgs;
    src=null;
    shellHook = ''
    pip install --user sphinx_bootstrap_theme
    export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.6/site-packages
    '';
  }
