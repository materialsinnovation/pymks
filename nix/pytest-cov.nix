{ nixpkgs, pypkgs }:
pypkgs.buildPythonPackage rec {
  pname = "pytest-cov";
  version = "2.5.1";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "03aa752cf11db41d281ea1d807d954c4eda35cfa1b21d6971966cc041bbf6e2d";
  };
  buildInputs = [
    pypkgs.pytest
    pypkgs.pytest_xdist
  ];
  ignoreCollisions=true;
  catchConflicts=false;
  propagatedBuildInputs = [ pypkgs.coverage ];
  doCheck = false;
}
