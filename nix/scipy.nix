{ nixpkgs, pypkgs }:
pypkgs.buildPythonPackage rec {
  pname = "scipy";
  version = "0.19.1";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "1rl411bvla6q7qfdb47fpdnyjhfgzl6smpha33n9ar1klykjr6m1";
  };
  buildInputs = [
    pypkgs.numpy
    nixpkgs.pkgs.gcc
    nixpkgs.pkgs.gfortran
  ];
  ignoreCollisions=true;
  catchConflicts=false;
  doCheck = false;
}
