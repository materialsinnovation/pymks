{ pypkgs, toml }:
pypkgs.buildPythonPackage rec {
  pname = "black";
  version = "18.6b4";
  src = pypkgs.fetchPypi {
    inherit pname version;
    sha256 = "0i4sfqgz6w15vd50kbhi7g7rifgqlf8yfr8y78rypd56q64qn592";
  };
  doCheck=false;
  buildInputs = [
    pypkgs.appdirs
    pypkgs.click
    pypkgs.attrs
    toml
  ];
  catchConflicts = false;
}
