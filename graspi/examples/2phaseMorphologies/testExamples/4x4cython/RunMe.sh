#!/bin/bash

MAINDIR=$PWD


# paths to GraSPI and external tools used to pre or post process data
GRASPI=$MAINDIR/../../../../src/graspiAPI

# file to analyze
FILENAME=data_4_3
# run GraSPI analysis
$GRASPI -a ${FILENAME}.txt -s 2 > descriptors-${FILENAME}-s2.txt 2>&1
