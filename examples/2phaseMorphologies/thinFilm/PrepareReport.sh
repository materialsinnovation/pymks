#!/bin/bash


PIXELSIZE=1

MAINDIR=$PWD
GRASPI=$MAINDIR/../../..

CONSTRHISTO=$GRASPI/tools/constructHistogram/ConstructHistogram1
CONSTRWHISTO=$GRASPI/tools/constructHistogram/ConstructWHistogram
AVG=$GRASPI/tools/statsTools/avg
VAR=$GRASPI/tools/statsTools/var


DATA="$MAINDIR/data"
DISTANCES="$MAINDIR/distances"
FIGS="$MAINDIR/figs"
HISTO="$MAINDIR/histograms"
DESCS="$MAINDIR/descriptors"
SRCDATA="$MAINDIR/src_data"
STATS="$MAINDIR/stats"


cd $DATA
TEXFILE=Report.tex

echo "\documentclass{article}" > $TEXFILE
echo "\usepackage{pslatex}" >> $TEXFILE
echo "\usepackage{amssymb}" >> $TEXFILE	
echo "\pagestyle{empty}" >> $TEXFILE
echo "\usepackage{graphicx}" >> $TEXFILE
echo "\usepackage{grffile}" >> $TEXFILE
echo "\usepackage{fullpage}" >> $TEXFILE
echo "\begin{document}" >> $TEXFILE


f=0;
for i in *.txt; do
    f=$(($f + 1))
    FILENAME=$i
    BASEFILENAME=`echo ${i} | sed 's/.txt//'` #remove txt-file extension
    echo ""
    echo "analyzing file $FILENAME"

    echo "" > $DESCS/descriptorsAdditional.$BASEFILENAME.log

    DINTANODE=DistancesGreenToRedViaBlack.txt
    DINTACATHODE=DistancesGreenToBlueViaWhite.txt
    DPOLCATHODE=DistancesBlackToRed.txt
    DFULLANODE=DistancesWhiteToBlue.txt
    DPOLINT=DistancesBlackToGreen.txt
    WDPOLINT=WDistancesBlackToGreen.txt
    TORTBLACKRED=TortuosityBlackToRed.txt
    TORTWHITEBLUE=TortuosityWhiteToBlue.txt

    MEANBLACKINT=0
    VARBLACKINT=0

    MEANINTBOTT=0
    VARINTBOTT=0
    
    MEANINTTOP=0
    VARINTTOP=0
    
    NOFLINES=`cat "$DISTANCES/${BASEFILENAME}-$DPOLINT" | wc -l`
    if [ $NOFLINES -gt 1 ]; then
	MEANBLACKINT=`cat $DISTANCES/${BASEFILENAME}-$DPOLINT | $AVG`
	VARBLACKINT=`cat $DISTANCES/${BASEFILENAME}-$DPOLINT  | $VAR`
    fi


    NOFLINES=`cat "$DISTANCES/${BASEFILENAME}-$DINTACATHODE" | wc -l`
    if [ $NOFLINES -gt 1 ]; then
	MEANINTBOTT=`cat $DISTANCES/${BASEFILENAME}-$DINTACATHODE | $AVG`
	VARINTBOTT=`cat $DISTANCES/${BASEFILENAME}-$DINTACATHODE  | $VAR`
    fi
    
    NOFLINES=`cat "$DISTANCES/${BASEFILENAME}-$DINTANODE" | wc -l`
    if [ $NOFLINES -gt 1 ]; then
	MEANINTTOP=`cat $DISTANCES/${BASEFILENAME}-$DINTANODE    | $AVG`
	VARINTTOP=`cat $DISTANCES/${BASEFILENAME}-$DINTANODE      | $VAR`
    fi

    STDDEVBLACKINT=`echo "scale=4; sqrt ($VARBLACKINT)" | bc -l` 
    STDDEVINTBOTT=`echo "scale=4; sqrt ($VARINTBOTT)" | bc -l` 
    STDDEVINTTOP=`echo "scale=4; sqrt ($VARINTTOP)" | bc -l` 


    echo "DISS_dist_D_Int \$\mu\$=$MEANBLACKINT , \$\sigma\$=$STDDEVBLACKINT \newline" >> $DESCS/descriptorsAdditional.$BASEFILENAME.log
    echo "CT_dist_Int_Ca_via_A \$\mu\$=$MEANINTBOTT , \$\sigma\$=$STDDEVINTBOTT \newline" >> $DESCS/descriptorsAdditional.$BASEFILENAME.log
    echo "CT_dist_Int_A_via_D \$\mu\$=$MEANINTTOP , \$\sigma\$=$STDDEVINTTOP \newline" >> $DESCS/descriptorsAdditional.$BASEFILENAME.log
    echo "Avg dev done!"

    $CONSTRHISTO $DISTANCES/${BASEFILENAME}-$DINTANODE 10
    mv Histogram.txt Histogram$DINTANODE
    mv CumHistogram.txt CumHistogram$DINTANODE
    $CONSTRHISTO $DISTANCES/${BASEFILENAME}-$DINTACATHODE 10
    mv Histogram.txt Histogram$DINTACATHODE
    mv CumHistogram.txt CumHistogram$DINTACATHODE
    $CONSTRHISTO $DISTANCES/${BASEFILENAME}-$DPOLCATHODE 10
    mv Histogram.txt Histogram$DPOLCATHODE
    mv CumHistogram.txt CumHistogram$DPOLCATHODE
    $CONSTRHISTO $DISTANCES/${BASEFILENAME}-$DFULLANODE 10
    mv Histogram.txt Histogram$DFULLANODE
    mv CumHistogram.txt CumHistogram$DFULLANODE
    $CONSTRHISTO $DISTANCES/${BASEFILENAME}-$DPOLINT 1
    mv Histogram.txt Histogram$DPOLINT
    mv CumHistogram.txt CumHistogram$DPOLINT
    $CONSTRWHISTO $DISTANCES/${BASEFILENAME}-$WDPOLINT 1
    mv WHistogram.txt WHistogram$WDPOLINT
    mv CumWHistogram.txt CumWHistogram$WDPOLINT
    $CONSTRHISTO $DISTANCES/${BASEFILENAME}-$TORTBLACKRED 0.02
    mv Histogram.txt Histogram$TORTBLACKRED 
    mv CumHistogram.txt CumHistogram$TORTBLACKRED 
    $CONSTRHISTO $DISTANCES/${BASEFILENAME}-$TORTWHITEBLUE 0.02 
    mv Histogram.txt Histogram$TORTWHITEBLUE 
    mv CumHistogram.txt CumHistogram$TORTWHITEBLUE

    # COLLECT STATS
    TOTV=`grep "STAT\_n "            $DESCS/descriptors.$BASEFILENAME.log | sed 's/STAT\_n //' `
    TOTVBLACK=`grep "STAT\_n\_D "    $DESCS/descriptors.$BASEFILENAME.log | sed 's/STAT\_n\_D //' `
    TOTVWHITE=`grep "STAT\_n\_A "    $DESCS/descriptors.$BASEFILENAME.log | sed 's/STAT\_n\_A //' `
    TOTINTCOMP=`grep "CT\_e\_conn "  $DESCS/descriptors.$BASEFILENAME.log | sed 's/CT\_e\_conn //' `

    echo "TOT=$TOTV" > Histogram.gp
    echo "TOTBLACK=$TOTVBLACK" >> Histogram.gp
    echo "TOTWHITE=$TOTVWHITE" >> Histogram.gp
    echo "TOTINTBLACKANDWHITE=$TOTINTCOMP" >> Histogram.gp
    cat $MAINDIR/Histograms_template.gp >> Histogram.gp

    gnuplot Histogram.gp


    # PREPARE TEX FILE
    BASEFILENAMEwoS=$(echo "$BASEFILENAME" | sed  's/_/\\_/g' )
    echo $BASEFILENAMEwoS
    echo -e "\section{Morphology: $BASEFILENAMEwoS }" >> $TEXFILE

    echo "\parbox{0.35\textwidth}{" >> $TEXFILE

    echo "\includegraphics[width=0.35\textwidth]{$FIGS/${BASEFILENAME}.png} \\  " >> $TEXFILE
    echo " ~\newline ~\newline " >> $TEXFILE

    echo "\begin{small}" >> $TEXFILE
#    cat $DESCS/descriptors.$BASEFILENAME.log  >> $TEXFILE

    cat $DESCS/descriptors.$BASEFILENAME.log | sed -e 's/$/\\\\/' |  sed -e 's/_/ /g'  >> $TEXFILE
    cat $DESCS/descriptorsAdditional.$BASEFILENAME.log | sed -e 's/_/ /g' >> $TEXFILE
    echo "\end{small}" >> $TEXFILE
    echo "}" >> $TEXFILE

    echo "\parbox{0.60\textwidth}{" >> $TEXFILE

    echo "\parbox{0.3\textwidth}{\centering Distance from A to Ca \newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramFullereneToCathode.pdf} \\ ~ \\ } " >> $TEXFILE
    echo "\parbox{0.3\textwidth}{\centering Distance from D to Am \newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramPolymerToAnode.pdf} \\ ~ \\ }" >> $TEXFILE
    echo "\parbox{0.3\textwidth}{\centering Path balance \newline " >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}BothHistogramsUsefulDomains.pdf} \\ ~ \\ }" >> $TEXFILE
    echo "\parbox{0.3\textwidth}{\centering Distance from D to Int \newline " >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramPolymerToInterface.pdf} \\ ~ \\ }" >> $TEXFILE
    echo "\parbox{0.3\textwidth}{\centering Tortuosity of D-paths to An  \newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramTortuosityPolymerToAnode.pdf} \\ ~ \\ }" >> $TEXFILE
    echo "\parbox{0.3\textwidth}{\centering Tortuosity of A-paths to Ca \newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramTortuosityFullereneToCathode.pdf} \\ ~ \\ }" >> $TEXFILE

    echo "}" >> $TEXFILE

    echo "\newpage" >> $TEXFILE
#    echo "~" >> $TEXFILE
#    echo "\newline" >> $TEXFILE



    #CLEAN
    for j in *Histogram*.txt; do
	mv $j $HISTO/${FILENAME}${j}
    done

    for j in *.eps; do
	mv $j $HISTO/${FILENAME}$j
    done


done

echo "\end{document}" >> $TEXFILE

cd $HISTO
for i in *.eps; do
	epstopdf $i
done

mv $DATA/$TEXFILE $MAINDIR

cd $MAINDIR
#pdflatex $TEXFILE
rm *.aux *.log

