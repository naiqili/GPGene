#!/bin/bash
for sizem in 10
do
    for cc in 50 
    do
        width=10
        data=data12
        maxiter=20000
        interv=10
        inpath=/home/linaiqi/Lab/data/gene/monocle/processed/
        outfile=/home/linaiqi/Lab/data/gene/res/monocle/monocle\_cc$cc\_m$sizem\_width$width\_interv$interv\_$data
        echo $outfile
        python main_monocle_ts.py --inpath $inpath --outfile $outfile --sizem $sizem --cc $cc --width $width --data $data --maxiter $maxiter --interv $interv --kernel Poly2
    done
done