{ nixpkgs, pypkgs }:
pypkgs.buildPythonPackage rec {
  name = "nbval-0.9.0";
  src = nixpkgs.fetchurl {
    url = "https://pypi.python.org/packages/58/ce/d705c865bdec10ab94c1c57b76e77f07241ef5c11c4976ec7e00de259f92/nbval-0.9.0.tar.gz";
    sha256 = "dec2a26bb32a29f92a577554b7142f960b8a91bca8cfaf23f4238718197bf7f3";
  };
  doCheck=false;
  buildInputs = [
    pypkgs.ipython
    pypkgs.jupyter_client
    pypkgs.tornado
    pypkgs.nbformat
    pypkgs.ipykernel
    pypkgs.coverage
    pypkgs.pytest
  ];
}
