let
  nixpkgs = import ./nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  pymks = import ../default.nix;
in
  nixpkgs.stdenv.mkDerivation rec {
    name = "pymks-dev";
    buildInputs = [
      pypkgs.virtualenv
      pypkgs.pip
      pymks
      pypkgs.ipdb
      pypkgs.snakeviz
    ] ++ pymks.buildInputs;
    src=null;
    shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    \rm -rf $HOME/.local
    mkdir -p $HOME/.local
    #pip install --user profile-viewer==0.1.5
    export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH
    export PATH=$PATH:$HOME/.local/bin
    '';
  }
