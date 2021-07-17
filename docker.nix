# View DOCKER.md to see how to use this

{
  tag ? "20.09",
  withSfepy ? true,
  withGraspi ? true,
  graspiVersion ? "59f6a8a2e1ca7c8744a4e37701b919131efb2f45"
}:
let
  pkgs = import (builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/${tag}.tar.gz") {};
  pypkgs = pkgs.python3Packages;
  pymks = pypkgs.callPackage ./default.nix {
    sfepy=(if withSfepy then pypkgs.sfepy else null);
    graspi=(if withGraspi then graspi else null);
  };
  graspisrc = builtins.fetchTarball "https://github.com/owodolab/graspi/archive/${graspiVersion}.tar.gz";
  graspi = pypkgs.callPackage "${graspisrc}/default.nix" {};

  lib = pkgs.lib;
  USER = "main";

  from_directory = dir: (lib.mapAttrsToList (name: type:
    if type == "directory" || (lib.hasSuffix "~" name) then
      null
    else
      dir + "/${name}"
  ) (builtins.readDir dir));

  ## files to copy into the user's home area in container
  files_to_copy = [ ] ++ (lib.remove null (from_directory ./notebooks));

  ## functions necessary to copy files to USER's home area
  ## is there an easier way???
  filetail = path: lib.last (builtins.split "(/)" (toString path));
  make_cmd = path: "cp ${path} ./home/${USER}/${filetail path}";
  copy_cmd = paths: builtins.concatStringsSep ";\n" (map make_cmd paths);

  python-env = pkgs.python3.buildEnv.override {
    ignoreCollisions = true;
    extraLibs = with pypkgs; [
      pymks
      ipywidgets
    ] ++ pymks.propagatedBuildInputs;
  };
in
  pkgs.dockerTools.buildImage {
    name = "wd15/pymks";
    tag = "latest";

    contents = [
      python-env
      pkgs.bash
      pkgs.busybox
      pkgs.coreutils
      pkgs.openssh
      pkgs.bashInteractive
    ];

    runAsRoot = ''
      #!${pkgs.stdenv.shell}
      ${pkgs.dockerTools.shadowSetup}
      groupadd --system --gid 65543 ${USER}
      useradd --system --uid 65543 --gid 65543 -d / -s /sbin/nologin ${USER}
    '';

    extraCommands = ''
      mkdir -m 1777 ./tmp
      mkdir -m 777 -p ./home/${USER}
      # echo 'extra commands'
      # pwd
      # cp -r notebooks ./home/${USER}/
    '' + copy_cmd files_to_copy;

    config = {
      Cmd = [ "bash" ];
      User = USER;
      Env = [
        "OMPI_MCA_plm_rsh_agent=${pkgs.openssh}/bin/ssh"
        "HOME=/home/${USER}"
      ];
      WorkingDir = "/home/${USER}";
      Expose = {
        "8888/tcp" = {};
      };
    };
  }
