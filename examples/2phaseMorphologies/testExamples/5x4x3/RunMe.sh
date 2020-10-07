#!/bin/bash

CDDIR=$PWD
MAINDIR=$CDDIR/../..

# paths to GraSPI and external tools used to pre or post process data
GRASPI=$MAINDIR/src/graspi

# file to analyze
FILENAME=data_5x4x3
# run GraSPI analysis
$GRASPI -a ${FILENAME}.txt > ${FILENAME}-a.log 2>&1

