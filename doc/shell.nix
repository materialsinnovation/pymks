let
  nixpkgs = import ../nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
in
  nixpkgs.stdenv.mkDerivation rec {
    name = "pymks-doc-env";
    buildInputs = [
      pypkgs.python
      pypkgs.sphinx
      pypkgs.virtualenv
      pypkgs.pip
    ];
    src=null;
    shellHook = ''
    pip install --user sphinx_bootstrap_theme
    export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.6/site-packages
    '';
  }
