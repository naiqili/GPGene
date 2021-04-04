import os
import pandas as pd


for neti in range(1, 6):
# for neti in range(1, 2):
    for sizen in range(100, 1001, 100):
        os.system('rm -rf ./inputs/LEAP/GNW/LEAP/*')
        with open('./inputs/LEAP/GNW/celldata.csv', 'w') as f:
            f.write('PseudoTime1\n')
            for i in range(sizen):
                f.write('E%d,%d\n'%(i,i))
        infile='/home/linaiqi/Lab/data/gene/mf100net%d.txt'%neti
        outfile='/home/linaiqi/Lab/data/gene/tmp/leap_mf100net%d_n%d.txt'%(neti, sizen)

        ex_matrix = pd.read_csv(infile, sep='\t')
        ex_matrix=ex_matrix[:sizen].transpose()
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