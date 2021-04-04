from GENIE3 import *
import argparse

parser = argparse.ArgumentParser(description='main.')
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--sizen', type=int)

args = parser.parse_args()

sizen = args.sizen

# fname='test.txt'
fname=args.infile

data0 = loadtxt(fname,skiprows=1)
f = open(fname)
gene_names = f.readline()
f.close()
gene_names = gene_names.rstrip('\n').split('\t')

data = data0[:sizen]
print(data.shape)

tree_method='RF'
# tree_method='ET'
# Number of randomly chosen candidate regulators at each node of a tree
K = 'sqrt'
# Number of trees per ensemble
ntrees = 1000
# Run the method with these settings
VIM3 = GENIE3(data,tree_method=tree_method,K=K,ntrees=ntrees)

res_str = get_link_list(VIM3,gene_names=gene_names,file_name=args.outfile)