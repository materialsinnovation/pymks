#!/bin/bash

CDDIR=$PWD

PLT2ARR=$CDDIR/../../src/plt2arr

for i in *.plt; do
	FILENAME=$i
	echo $FILENAME
	BASEFILENAME=`echo ${FILENAME} | sed 's/.plt//'`
	$PLT2ARR $FILENAME ${BASEFILENAME}.txt
done

