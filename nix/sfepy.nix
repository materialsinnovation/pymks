{ nixpkgs, pypkgs }:
pypkgs.buildPythonPackage rec {
  name = "sfepy-2017.4.1";
  src = nixpkgs.fetchurl {
    url = "https://github.com/sfepy/sfepy/archive/release_2017.4.1.tar.gz";
    sha256 = "06x3xnfxz6iwp3wd926298ybb4p707bciwkq5f13sfqwc9m16km8";
  };
  doCheck = false;
  buildInputs = [
    pypkgs.numpy
    pypkgs.cython
    pypkgs.scipy
    pypkgs.matplotlib
    pypkgs.pyparsing
    pypkgs.tables
  ];
  catchConflicts = false;
}
