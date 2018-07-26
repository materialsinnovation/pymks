{ pypkgs }:
pypkgs.buildPythonPackage rec {
  pname = "toml";
  version = "0.9.4";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "0bdbpbip67wdm6c7xwc6mmbmskyradj4cdxn1iibj4fcx1nbv1lf";
  };
  doCheck=false;
  buildInputs = [
  ];
}
