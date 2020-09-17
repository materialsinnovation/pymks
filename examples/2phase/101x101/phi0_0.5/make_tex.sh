#!/bin/bash

MUH=1
MUE=1
PIXELSIZE=1

GRASPI=/home/owodo/MINE/PROJECTS/GraSPI/GraSPI-2.0
GRASPIEXEC=$GRASPI/src/graspi
CONSTRHISTO=$GRASPI/tools/constructHistogram/ConstructHistogram

MAINDIR=$PWD

SRCDATA="$MAINDIR/src_data"
STATS="$MAINDIR/stats"
LOGS="$MAINDIR/logs"
FIGS="$MAINDIR/figs"
HISTO="$MAINDIR/histograms"
DISTANCES="$MAINDIR/distances"

cd $SRCDATA
TEXFILE=Report.tex

echo "\documentclass{article}" > $TEXFILE
echo "\usepackage{pslatex}" >> $TEXFILE
echo "\usepackage{amssymb}" >> $TEXFILE	
echo "\pagestyle{empty}" >> $TEXFILE
echo "\usepackage{graphicx}" >> $TEXFILE
echo "\usepackage{grffile}" >> $TEXFILE
echo "\usepackage{fullpage}" >> $TEXFILE
echo "\begin{document}" >> $TEXFILE

echo "" > $STATS/AllEta.dat

f=0;
for i in phasedataCH.0004.txt; do
    f=$(($f + 1))
    FILENAME=`basename $i`
    BASEFILENAME=`echo ${i} | sed 's/.txt//'` #remove txt-file extension
    echo ""
    echo "analyzing file $FILENAME"

    $GRASPIEXEC -a $FILENAME -s $PIXELSIZE > $LOGS/graspi-tmp.$BASEFILENAME.log

    cat $LOGS/graspi-tmp.$BASEFILENAME.log \
	| sed 's/white/fullerene/' | sed 's/black/polymer/' \
	| sed 's/red/anode/' | sed 's/blue/cathode/' \
	| sed 's/green/interface/' > $LOGS/graspi.$BASEFILENAME.log

    # COLLECT STATS
    STR="\[STATS\] Number of vertices: "
    TOTV=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[STATS\] Number of black vertices: "
    TOTVBLACK=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[STATS\] Number of white vertices: "
    TOTVWHITE=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[STATS\] Number of black int vertices with path to red: "
    TOTVBLACKINT=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[STATS\] Number of white int vertices with path to blue: "
    TOTVWHITEINT=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[STATS\] Number of int edges with complementary paths: "
    TOTINTCOMP=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    TOTINTBLACKANDWHITE=$(echo "$TOTINTCOMP*2" | bc)

    STR="\[ETA ABS\] Fraction of black vertices: "
    ETAABSUPP=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA ABS\] Weighted fraction of black vertices: "
    ETAABSLOW=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA CT\] Fraction of useful vertices \- w\/o islands: "
    ETAOUTUPP=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA CT\] Fraction of black vertices conn to red: "
    ETAOUTUPPPOL=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA CT\] Fraction of white vertices conn to blue: "
    ETAOUTUPPFULL=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA CT\] Fraction of interface with complementary paths to blue and red: "
    ETAOUTPATH=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA DISS\] Number of green 1st order edges: "
    INTAREA=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA DISS\] Weighted fraction of black vertices in 10 distance to green: "
    ETADISSWLD10=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA CT\] Fraction of black vertices with straight rising paths (t=1): "
    ETAOUTTORTPOL=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `
    STR="\[ETA CT\] Fraction of white vertices with straight rising paths (t=1): "
    ETAOUTTORTFULL=`grep -e "$STR" $LOGS/graspi-tmp.$BASEFILENAME.log | sed -e "s/$STR//" `


    BLACKUSE=`echo "scale=8; $ETAOUTUPPPOL*$TOTVBLACK" | bc -l` 
    WHITEUSE=`echo "scale=8; $ETAOUTUPPFULL*$TOTVWHITE" | bc -l` 
    echo "Stats collected!"

    echo "$BASEFILENAME $ETAABSUPP $ETAABSLOW $INTAREA $ETADISSWLD10 $ETAOUTUPP $ETAOUTUPPPOL $ETAOUTUPPFULL $ETAOUTTORTPOL $ETAOUTTORTFULL " >> $STATS/AllEta.dat
#    echo "$BASEFILENAME $ETAABSUPP $ETAOUTUPPPOL $ETAOUTUPPFULL $ETAOUTPATH $INTAREA 0$FOFLD 0$WFOFLD 0$TORTBLACK 0$TORTWHITE" >> AllEta.dat


    DINTANODE=DistancesGreenToRedViaBlack.txt
    DINTACATHODE=DistancesGreenToBlueViaWhite.txt
    DPOLCATHODE=DistancesBlackToRed.txt
    DFULLANODE=DistancesWhiteToBlue.txt
    DPOLINT=DistancesBlackToGreen.txt
    TORTBLACKRED=TortuosityBlackToRed.txt
    TORTWHITEBLUE=TortuosityWhiteToBlue.txt

    $CONSTRHISTO $DINTANODE 10
    mv Histogram.txt Histogram$DINTANODE
    mv CumHistogram.txt CumHistogram$DINTANODE
    $CONSTRHISTO $DINTACATHODE 10
    mv Histogram.txt Histogram$DINTACATHODE
    mv CumHistogram.txt CumHistogram$DINTACATHODE
    $CONSTRHISTO $DPOLCATHODE 10
    mv Histogram.txt Histogram$DPOLCATHODE
    mv CumHistogram.txt CumHistogram$DPOLCATHODE
    $CONSTRHISTO $DFULLANODE 10
    mv Histogram.txt Histogram$DFULLANODE
    mv CumHistogram.txt CumHistogram$DFULLANODE
    $CONSTRHISTO $DPOLINT 1
    mv Histogram.txt Histogram$DPOLINT
    mv CumHistogram.txt CumHistogram$DPOLINT
    $CONSTRHISTO $TORTBLACKRED 0.1
    mv Histogram.txt Histogram$TORTBLACKRED
    mv CumHistogram.txt CumHistogram$TORTBLACKRED
    $CONSTRHISTO $TORTWHITEBLUE 0.1
    mv Histogram.txt Histogram$TORTWHITEBLUE
    mv CumHistogram.txt CumHistogram$TORTWHITEBLUE

    echo "TOT=$TOTV" > Histogram.gp
    echo "TOTBLACK=$TOTVBLACK" >> Histogram.gp
    echo "TOTWHITE=$TOTVWHITE" >> Histogram.gp
    echo "TOTBLACKINT=$TOTVBLACKINT" >> Histogram.gp
    echo "TOTWHITEINT=$TOTVWHITEINT" >> Histogram.gp
    echo "TOTINTBLACKANDWHITE=$TOTINTBLACKANDWHITE" >> Histogram.gp
    echo "BLACKUSE=$BLACKUSE" >> Histogram.gp
    echo "WHITEUSE=$WHITEUSE" >> Histogram.gp
    cat $MAINDIR/Histograms_template.gp >> Histogram.gp

    gnuplot Histogram.gp


    # PREPARE TEX FILE
    echo -e "\section{Morphology: $BASEFILENAME }" >> $TEXFILE
    echo "\begin{small}" >> $TEXFILE
    echo "\begin{center}\includegraphics[width=0.2\textwidth]{$FIGS/$BASEFILENAME.plt.jpg} \end{center}" >> $TEXFILE
    cat $LOGS/graspi.$BASEFILENAME.log | sed -e 's/$/\\\\/' | sed -e 's/\[//' | sed -e 's/\]/ | /'  >> $TEXFILE
    echo "" >> $TEXFILE
    echo "\end{small}" >> $TEXFILE

    echo "\begin{center}" >> $TEXFILE
    echo "\parbox{0.33\textwidth}{\begin{scriptsize}Distance from Fullerene Vertex to Cathode\end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}CumHistogramFullereneToCathode.pdf}} " >> $TEXFILE
    echo "\parbox{0.33\textwidth}{\begin{scriptsize}Distance from Polymer Vertex to Anode\end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}CumHistogramPolymerToAnode.pdf}}" >> $TEXFILE
    echo "\parbox{0.33\textwidth}{\begin{scriptsize}Distance from Polymer Vertex to Interface\end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}CumHistogramPolymerToInterface.pdf}} \newline" >> $TEXFILE

    echo "\parbox{0.49\textwidth}{\begin{scriptsize}Distance from Interface to Cathode via Fullerene vertices \end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}CumHistogramInterfaceToCathodeViaFullerene.pdf}}" >> $TEXFILE
    echo "\parbox{0.49\textwidth}{\begin{scriptsize}Distance from Interface to Anode via Polymer vertices \end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}CumHistogramInterfaceToAnodeViaPolymer.pdf}}" >> $TEXFILE

    echo "\parbox{0.33\textwidth}{\begin{scriptsize}Distance from Fullerene Vertex to Cathode\end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramFullereneToCathode.pdf}} " >> $TEXFILE
    echo "\parbox{0.33\textwidth}{\begin{scriptsize}Distance from Polymer Vertex to Anode\end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramPolymerToAnode.pdf}}" >> $TEXFILE
    echo "\parbox{0.33\textwidth}{\begin{scriptsize}Distance from Polymer Vertex to Interface\end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.3\textwidth]{$HISTO/${FILENAME}HistogramPolymerToInterface.pdf}} \newline" >> $TEXFILE

    echo "\parbox{0.49\textwidth}{\begin{scriptsize}Distance from Interface to Cathode via Fullerene vertices \end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}HistogramInterfaceToCathodeViaFullerene.pdf}}" >> $TEXFILE
    echo "\parbox{0.49\textwidth}{\begin{scriptsize}Distance from Interface to Anode via Polymer vertices \end{scriptsize}\newline" >> $TEXFILE
    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}HistogramInterfaceToAnodeViaPolymer.pdf}}\newline" >> $TEXFILE

    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}BothHistogramsUsefulDomains.pdf}" >> $TEXFILE
    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}BothHistogramsUsefulInterface.pdf}" >> $TEXFILE

    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}CumHistogramTortuosityPolymerToAnode.pdf}" >> $TEXFILE
    echo "\includegraphics[width=0.33\textwidth]{$HISTO/${FILENAME}CumHistogramTortuosityFullereneToCathode.pdf}" >> $TEXFILE

    echo "\end{center}" >> $TEXFILE
    #CLEAN
    for j in *Histogram*.txt; do
	mv $j $HISTO/${FILENAME}${j}
    done

    for j in Distances*.txt; do
	mv $j $DISTANCES/${BASEFILENAME}-${j}
    done
    for j in Tortuosity*.txt; do
	mv $j $DISTANCES/${BASEFILENAME}-${j}
    done

    for j in *.eps; do
	mv $j $HISTO/${FILENAME}$j
    done


    if [ -f $BASEFILENAME.graphe ]; then
	rm $BASEFILENAME.graphe
    fi

done
echo "\end{document}" >> $TEXFILE

cd $HISTO
for i in *.eps; do
	epstopdf $i
done

mv $SRCDATA/$TEXFILE $MAINDIR

cd $MAINDIR
pdflatex $TEXFILE
rm *.aux *.log

