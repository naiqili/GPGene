#!/bin/bash
for sizem in 40
do
    #for cc in 10 50 100
    for cc in 50
    do
        echo "sizem $sizem"
        infile=/home/linaiqi/Lab/data/gene/ts10gene201.txt
        outfile=/home/linaiqi/Lab/data/gene/res/mf10gene/gpgene\_10nw\_cc$cc\_m$sizem
        echo $outfile
        python main_ts.py --infile $infile --outfile $outfile --sizem $sizem --cc $cc --gene 10 --kernel Poly2
    done
done