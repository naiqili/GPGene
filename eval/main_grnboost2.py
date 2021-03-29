import os
import pandas as pd

from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names

import multiprocessing

import argparse

if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='main.')
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--sizen', type=int)

    args = parser.parse_args()

    net1_ex_path = args.infile

    ex_matrix = pd.read_csv(net1_ex_path, sep='\t')
    ex_matrix=ex_matrix[:args.sizen]
    tf_names = list(ex_matrix.columns)

    network = grnboost2(expression_data=ex_matrix,
                        tf_names=tf_names)

    network.to_csv(args.outfile, sep='\t', header=False, index=False)