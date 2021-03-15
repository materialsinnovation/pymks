{ lib
, buildPythonPackage
, fetchPypi
, pytestCheckHook
, sfepy
, dask
, scikitlearn
, pytestcov
, deprecated
, nbval
, dask-ml
, python
, matplotlib
, openssh
, graphviz
, pygraphviz
, pyfftw
}:
buildPythonPackage rec {
  pname = "pymks";
  version = "0.4.dev";

  src = builtins.filterSource (path: type: type != "directory" || baseNameOf path != ".git") ./.;

  propagatedBuildInputs = [
    sfepy
    dask
    scikitlearn
    pytestcov
    deprecated
    nbval
    dask-ml
    matplotlib
    openssh
    graphviz
    pygraphviz
    pyfftw
  ];

  checkInputs = [
    python
  ];

  checkPhase = ''
    export OMPI_MCA_plm_rsh_agent=${openssh}/bin/ssh
    ${python.interpreter} -c "import pymks; pymks.test()"
  '';

  pythonImportsCheck = ["pymks"];

  meta = with lib; {
    homepage = "https://pymks.org";
    description = "The Materials Knowledge System in Python";
    license = licenses.mit;
    maintainers = with maintainers; [ wd15 ];
  };
}
