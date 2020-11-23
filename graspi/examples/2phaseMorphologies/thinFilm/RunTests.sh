#!/bin/bash

MAINDIR=$PWD


PIXELSIZE=1

GRASPI=$MAINDIR/../../..
GRASPIEXEC=$GRASPI/src/graspi


DATA="$MAINDIR/data"
DISTANCES="$MAINDIR/distances"
FIGS="$MAINDIR/figs"
HISTO="$MAINDIR/histograms"
DESCS="$MAINDIR/descriptors"
STATS="$MAINDIR/stats"
SRCDATA="$MAINDIR/src_data"

cd $DATA

f=0;
for i in *.txt; do
    f=$(($f + 1))
    FILENAME=$i
    BASEFILENAME=`echo ${i} | sed 's/.txt//'` #remove txt-file extension
    echo ""
    echo "analyzing file $FILENAME"
    $GRASPIEXEC -a $FILENAME -s $PIXELSIZE -p 1 > $DESCS/descriptors.$BASEFILENAME.log

    for j in *Distances*.txt; do
	mv $j $DISTANCES/${BASEFILENAME}-${j}
    done

    for j in *CC*.txt; do
	mv $j $DISTANCES/${BASEFILENAME}-${j}
    done


    for j in *Tortuosity*.txt; do
	mv $j $DISTANCES/${BASEFILENAME}-${j}
    done

done

cd $MAINDIR

