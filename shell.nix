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
  });

  pymks = pypkgs.callPackage ./default.nix {
    sfepy=(if withSfepy then sfepy else null);
    graspi=(if withGraspi then graspi else null);
  };
  extra = with pypkgs; [ black pylint flake8 ipywidgets ];
  graspisrc = builtins.fetchTarball "https://github.com/owodolab/graspi/archive/${graspiVersion}.tar.gz";
  graspi = pypkgs.callPackage "${graspisrc}/default.nix" {};
in
  (pymks.overridePythonAttrs (old: rec {

    propagatedBuildInputs = old.propagatedBuildInputs;

    nativeBuildInputs = propagatedBuildInputs ++ extra;

    postShellHook = ''
      export OMPI_MCA_plm_rsh_agent=${pkgs.openssh}/bin/ssh

      SOURCE_DATE_EPOCH=$(date +%s)
      export PYTHONUSERBASE=$PWD/.local
      export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
      export PYTHONPATH=$PYTHONPATH:$USER_SITE:$(pwd)
      export PATH=$PATH:$PYTHONUSERBASE/bin

      jupyter nbextension install --py widgetsnbextension --user > /dev/null 2>&1
      jupyter nbextension enable widgetsnbextension --user --py > /dev/null 2>&1
      pip install jupyter_contrib_nbextensions --user > /dev/null 2>&1
      jupyter contrib nbextension install --user > /dev/null 2>&1
      jupyter nbextension enable spellchecker/main > /dev/null 2>&1
    '';
  }))
