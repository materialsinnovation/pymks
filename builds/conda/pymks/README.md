# Building and uploading to Conda pymks/pymks channel

The user and channel is, https://anaconda.org/pymks/pymks.

Remember to change the `version` and `git_rev`.

To build and upload use,

    $ conda update -n root conda-build
    $ conda clean --lock
    $ conda clean --all
    $ conda build --python=3.5
    $ anaconda upload -u pymks -c pymks /home/wd15/anaconda/conda-bld/linux-64/pymks-0.3.3.dev17+g817dc61-py35_1.tar.bz2
