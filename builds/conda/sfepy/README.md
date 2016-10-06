Unable to build see issue https://github.com/materialsinnovation/pymks/issues/323.

For reference, the files are stored at
https://anaconda.org/pymks/pymks. Useful commands when building with
conda are below. They fixed issues with the build up until the failure
in the issue above.

    $ conda update -n root conda-build
    $ conda clean --all
    $ conda build --python=3.5 .
    $ anaconda upload home/wd15/anaconda/conda-bld/linux-64/sfepyXXXXX
