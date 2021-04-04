import os
import pandas as pd

with open('./inputs/LEAP/GNW/celldata.csv', 'w') as f:
    f.write('PseudoTime1\n')
    for i in range(200):
        f.write('E%d,%d\n'%(i,i))

os.system('rm -rf ./inputs/LEAP/GNW/LEAP/*')
# infile='/home/linaiqi/Lab/GPGene-TCBB/data/synthetic/synnet_and_data.txt'
infile='/home/linaiqi/Lab/GPGene-TCBB/data/synthetic/synnet_andnot_data.txt'
outfile='./out.txt'

ex_matrix = pd.read_csv(infile, sep='\t')
ex_matrix=ex_matrix.transpose()
ex_matrix=ex_matrix.add_prefix('E')
tf_names = list(ex_matrix.columns)

tmp_file='./inputs/LEAP/GNW/tmp.csv'
ex_matrix.to_csv(tmp_file)

os.system('python BLRunner.py --config leap_config.yaml')

with open('./outputs/LEAP/GNW/LEAP/rankedEdges.csv') as fin, open(outfile, 'w') as fout:
    fin.readline()
    for l in fin:
        g1, g2, v = l.strip().split('\t')
        if g1!=g2:
            fout.write('%s\t%s\t%s\n'%(g1,g2,v))