{ nixpkgs ? import <nixpkgs> {} }:
let
  python36Packages = nixpkgs.python36Packages;
in
  python36Packages.buildPythonPackage rec {
  name = "nbval-0.9.0";
  src = nixpkgs.fetchurl {
    url = "https://pypi.python.org/packages/58/ce/d705c865bdec10ab94c1c57b76e77f07241ef5c11c4976ec7e00de259f92/nbval-0.9.0.tar.gz";
    sha256 = "dec2a26bb32a29f92a577554b7142f960b8a91bca8cfaf23f4238718197bf7f3";
  };
  doCheck=false;
  buildInputs = [
    python36Packages.ipython
    python36Packages.jupyter_client
    python36Packages.tornado
    python36Packages.nbformat
    python36Packages.ipykernel
    python36Packages.coverage
    python36Packages.pytest
  ];
}
