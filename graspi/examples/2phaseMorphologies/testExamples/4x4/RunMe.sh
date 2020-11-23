#!/bin/bash

MAINDIR=$PWD


# paths to GraSPI and external tools used to pre or post process data
GRASPI=$MAINDIR/../../../../src/graspi

# file to analyze
FILENAME=data_4_3
# run GraSPI analysis
$GRASPI -a ${FILENAME}.txt -s 2 > descriptors-${FILENAME}-s2.txt 2>&1
#$GRASPI -a ${FILENAME}.txt -s 2 -p 1 > descriptors-${FILENAME}-s2p1.log 2>&1

#$GRASPI -g ${FILENAME}.graphe > descriptors-${FILENAME}-g.log 2>&1

#$GRASPI -a ${FILENAME}.txt -r /home/owodo/Desktop/ > descriptors-${FILENAME}-a.log 2>&1

# file to analyze
FILENAME=data_4_3_2
# run GraSPI analysis
#$GRASPI -a ${FILENAME}.txt -s 2 > descriptors-${FILENAME}-s2.log 2>&1
#$GRASPI -a ${FILENAME}.txt -s 2 -p 1 > descriptors-${FILENAME}-s2p1.log 2>&1

#$GRASPI -g ${FILENAME}.graphe > descriptors-${FILENAME}-g.log 2>&1
#$GRASPI -a ${FILENAME}.txt > descriptors-${FILENAME}-a.log 2>&1
