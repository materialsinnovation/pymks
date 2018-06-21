{ pypkgs }:
let
  python = pypkgs.python;
in
  pypkgs.scikitlearn.overridePythonAttrs (oldAttrs: {checkPhase=''
    HOME=$TMPDIR OMP_NUM_THREADS=1 nosetests --doctest-options=+SKIP $out/${python.sitePackages}/sklearn/
  '';})
