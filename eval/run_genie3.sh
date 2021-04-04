#!/bin/bash
for i in `seq 1 5`;
do
    echo "net$i"
    for sizen in 100 200 300 400 500 600 700 800 900 1000
    do
        echo "sizen $sizen"
        infile=/home/linaiqi/Lab/data/gene/mf100net$i.txt
        outfile=/home/linaiqi/Lab/data/gene/tmp/genie3_mf100net$i\_n$sizen.txt
        echo $outfile
        python main.py --infile $infile --outfile $outfile --sizen $sizen
    done
done