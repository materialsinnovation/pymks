{ pypkgs, dask-searchcv, dask-glm }:
pypkgs.buildPythonPackage rec {
  pname = "dask-ml";
  version = "0.6.0";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "0y5fw3iagfm2kjcr1fwnansyww0qfr98bw1fv0810hsflx9cqnsj";
  };
  buildInputs = [
    pypkgs.numpy
    pypkgs.pandas
    pypkgs.scikitlearn
    dask-searchcv
    pypkgs.multipledispatch
    dask-glm
    pypkgs.dask
    pypkgs.setuptools_scm
  ];
  ignoreCollisions=true;
  catchConflicts=false;
  doCheck = false;
}
