let
  inherit (import <nixpkgs> {}) fetchFromGitHub;
  nixpkgs_download = fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs-channels";
    rev = "1bc5bf4beb759e563ffc7a8a3067f10a00b45a7d";
    sha256 = "00gd96p7yz3rgpjjkizp397y2syfc272yvwxqixbjd1qdshbizmj";
  };
in
  import nixpkgs_download {}
