#!/bin/bash

fout=/home/linaiqi/Lab/data/gene/tmp/result\_gpgeneP2\_100gene.txt
rm -f $fout

unset DISPLAY

for i in `seq 1 5`;
do
    for sizen in 100 200 300 400 500 600 700 800 900 1000
    do
        echo -n "gpgeneL net $i sizen $sizen " >> $fout
        gsfile=/home/linaiqi/Lab/data/gene/gene100net$i.xml
        predfile=/home/linaiqi/Lab/data/gene/tmp/gpgene_mf100net$i\_n$sizen\_m40\_poly2.txt
        echo "java -jar gnw3-standalone.jar --evaluate --goldstandard $gsfile --prediction $predfile"
        java -jar gnw3-standalone.jar --evaluate --goldstandard $gsfile --prediction $predfile 2>&1 | grep 'AUPR' >> $fout
    done
done