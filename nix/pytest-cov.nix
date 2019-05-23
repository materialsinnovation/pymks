{ nixpkgs, pypkgs }:
pypkgs.buildPythonPackage rec {
  pname = "pytest-cov";
  version = "2.7.1";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "0filvmmyqm715azsl09ql8hy2x7h286n6d8z5x42a1wpvvys83p0";
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
