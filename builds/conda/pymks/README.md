# Building and uploading to Conda pymks/pymks channel

The user and channel is, https://anaconda.org/pymks/pymks.

Remember to change the `version` and `git_rev`.

To build and upload use,

    $ conda install -n root conda-build
    $ conda update -n root conda-build
    $ conda clean --lock # can help when can't build
    $ conda clean --all # can help when can't build
    $ conda build purge
    $ conda build --python=3.6 .
    $ anaconda upload -u pymks -c pymks /home/wd15/miniconda/conda-bld/linux-64/pymks-X.Y.Z.tar.bz2
