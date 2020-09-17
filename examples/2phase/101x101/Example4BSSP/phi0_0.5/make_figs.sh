#!/bin/bash

CDDIR=$PWD

DATA=src_data
FIGS=figs

if [ ! -d "$FIGS" ]; then
	mkdir $FIGS;
fi 


cp contmapplt2jpg $DATA
cd $DATA
./contmapplt2jpg
rm contmapplt2jpg
rm *.mcr
mv *.jpg $CDDIR/$FIGS
cd ..


