{ pypkgs }:
pypkgs.buildPythonPackage rec {
  pname = "dask-searchcv";
  version = "0.2.0";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "0rc5d9alcnsxf720jwi1ib1cfba0hgl540smb5bd5h5d3lksksn1";
  };
  buildInputs = [
    pypkgs.toolz
    pypkgs.numpy
    pypkgs.scikitlearn
    pypkgs.dask
  ];
  ignoreCollisions=true;
  catchConflicts=false;
  doCheck = false;
}
