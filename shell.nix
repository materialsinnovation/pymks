#
# $ nix-shell --pure --arg withBoost false --argstr tag 20.09
#

{
  tag ? "22.05",
  withSfepy ? true,
  withGraspi ? true,
  graspiVersion ? "59f6a8a2e1ca7c8744a4e37701b919131efb2f45"
}:
let
  pkgs = import (builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/${tag}.tar.gz") {};
  pypkgs = pkgs.python3Packages;
  sfepy = pypkgs.sfepy.overridePythonAttrs (old: rec {
    version = "2022.1";
    src = pkgs.fetchFromGitHub {
      owner = "sfepy";
       repo = "sfepy";
       rev = "release_${version}";
      sha256 = "sha256-OayULh/dGI5sEynYMc+JLwUd67zEGdIGEKo6CTOdZS8=";
    };
    meta = old.meta // { broken = false; };
  });
  pymks = pypkgs.callPackage ./default.nix {
    sfepy=(if withSfepy then sfepy else null);
    graspi=(if withGraspi then graspi else null);
  };
  extra = with pypkgs; [ black pylint flake8 ipywidgets ];
  graspisrc = builtins.fetchTarball "https://github.com/owodolab/graspi/archive/${graspiVersion}.tar.gz";
  graspi = pypkgs.callPackage "${graspisrc}/default.nix" {};
  nixes_src = builtins.fetchTarball "https://github.com/wd15/nixes/archive/9a757526887dfd56c6665290b902f93c422fd6b1.zip";
  jupyter_extra = pypkgs.callPackage "${nixes_src}/jupyter/default.nix" {
    jupyterlab=(if pkgs.stdenv.isDarwin then pypkgs.jupyter else pypkgs.jupyterlab);
  };

in
  (pymks.overridePythonAttrs (old: rec {

    propagatedBuildInputs = old.propagatedBuildInputs;

    nativeBuildInputs = propagatedBuildInputs ++ extra ++ [ jupyter_extra ];

    postShellHook = ''
      export OMPI_MCA_plm_rsh_agent=${pkgs.openssh}/bin/ssh

      SOURCE_DATE_EPOCH=$(date +%s)
      export PYTHONUSERBASE=$PWD/.local
      export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
      export PYTHONPATH=$PYTHONPATH:$USER_SITE:$(pwd)
      export PATH=$PATH:$PYTHONUSERBASE/bin

    '';
  }))
