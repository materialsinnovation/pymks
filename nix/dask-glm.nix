{ pypkgs, scipy }:
pypkgs.buildPythonPackage rec {
  pname = "dask-glm";
  version = "0.1.0";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "16nq5r1zqs7vdr4vpbly3bsxz9ch1bnpxmkw8niyd3sm71sx2f2s";
  };
  buildInputs = [
    scipy
    pypkgs.cloudpickle
    pypkgs.multipledispatch
    pypkgs.scikitlearn
    pypkgs.dask
    pypkgs.setuptools_scm
  ];
  ignoreCollisions=true;
  catchConflicts=false;
  doCheck = false;
}
