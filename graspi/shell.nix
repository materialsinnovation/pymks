{ pkgs ? (import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/20.09.tar.gz";
    sha256 = "1wg61h4gndm3vcprdcg7rc4s1v3jkm5xd7lw8r2f67w502y94gcy";
  }) {}) }:
let
  pypkgs = pkgs.python3Packages;
  graspi = pypkgs.buildPythonPackage rec {
    pname = "graspi";
    version = "0.1";
    src = builtins.filterSource (path: type: type != "directory" || baseNameOf path != ".git") ./.;
    buildInputs = with pypkgs; [
      cython
      pkgs.boost
      numpy
    ];
  };
in
  pkgs.mkShell rec {
    pname = "graspi-env";
    nativeBuildInputs = with pypkgs; [
      pkgs.boost
      graspi
      numpy
    ];
  }
