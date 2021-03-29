# Based on Beeline framework

import os
import pandas as pd

# with open('./inputs/pidc/GNW/celldata.csv', 'w') as f:
#     f.write('PseudoTime1,PseudoTime2\n')
#     for i in range(1, 1001):
#         f.write('%d,%d,NA\n'%(i,i))

for neti in range(1, 6):
# for neti in range(1, 2):
    for sizen in range(100, 1001, 100):
        os.system('rm ./inputs/pidc/GNW/PIDC/*')
        infile='/home/linaiqi/Lab/data/gene/mf100net%d.txt'%neti
        outfile='/home/linaiqi/Lab/data/gene/tmp/pidc_mf100net%d_n%d.txt'%(neti, sizen)

        ex_matrix = pd.read_csv(infile, sep='\t')
        ex_matrix=ex_matrix[:sizen].transpose()
        tf_names = list(ex_matrix.columns)

        tmp_file='./inputs/pidc/GNW/tmp.csv'
        ex_matrix.to_csv(tmp_file)
        
        os.system('python BLRunner.py --config pidc_config.yaml')
        
#         cmd='cp ./outputs/pidc/GNW/PIDC/outFile.txt %s'%outfile
#         print(cmd)
#         os.system(cmd)

        with open('./outputs/pidc/GNW/PIDC/outFile.txt') as fin, open(outfile, 'w') as fout:
            for l in fin:
                g1, g2, na = l.strip().split('\t')
                fout.write('%s\t%s\t0\n'%(g1,g2))