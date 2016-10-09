#!/bin/bash

## Following is a hack to build Sfepy as Conda build is broken
pwd=`pwd`
cd ..
git clone https://github.com/sfepy/sfepy.git
cd sfepy
git checkout release_2016.3
python setup.py install
cd ${pwd}
## End of Sfepy hack

$PYTHON setup.py install
