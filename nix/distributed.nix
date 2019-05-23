{ pypkgs }:
pypkgs.distributed.overridePythonAttrs (oldAttrs: {
  version = "1.25.3";
  pname = "distributed";
  src=pypkgs.fetchPypi {
    pname = "distributed";
    version = "1.25.3";
    sha256 = "0bvjlw74n0l4rgzhm876f66f7y6j09744i5h3iwlng2jwzyw97gs";
  };
  propagatedBuildInputs = [
    pypkgs.click pypkgs.cloudpickle pypkgs.dask pypkgs.msgpack pypkgs.psutil pypkgs.six
    pypkgs.sortedcontainers pypkgs.tblib pypkgs.toolz pypkgs.tornado pypkgs.zict pypkgs.pyyaml pypkgs.mpi4py pypkgs.bokeh];
})
