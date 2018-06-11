{ nixpkgs ? import <nixpkgs> {} }:
let
  python36Packages = nixpkgs.python36Packages;
in
  python36Packages.buildPythonPackage rec {
    pname = "pytest-cov";
    version = "2.5.1";
    src = python36Packages.fetchPypi {
      inherit pname version;
      sha256 = "03aa752cf11db41d281ea1d807d954c4eda35cfa1b21d6971966cc041bbf6e2d";
    };
    buildInputs = [
      python36Packages.pytest
      python36Packages.pytest_xdist
    ];
    propagatedBuildInputs = [ python36Packages.coverage ];
    doCheck = false;
  }
