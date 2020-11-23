#!/bin/bash

CDDIR=$PWD

PLT2ARR=$CDDIR/../../src/plt2arr

cd phi0_0.5
for i in phase*.plt; do
	FILENAME=$i
	echo $FILENAME
	BASEFILENAME=`echo ${FILENAME} | sed 's/.plt//'`
	$PLT2ARR $FILENAME ${BASEFILENAME}.txt
done
cd ..

cd phi0_0.63
for i in phase*.plt; do
	FILENAME=$i
	echo $FILENAME
	BASEFILENAME=`echo ${FILENAME} | sed 's/.plt//'`
	$PLT2ARR $FILENAME ${BASEFILENAME}.txt
done
cd ..
