from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # using specific GPU
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from compatible.likelihoods import MultiClass, Gaussian
from compatible.kernels import RBF, White
from gpflow.models.svgp import SVGP
from gpflow.training import AdamOptimizer, ScipyOptimizer
from scipy.stats import mode
from scipy.cluster.vq import kmeans2
import gpflow
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import autoflow, params_as_tensors, ParamList
import pandas as pd
import itertools
pd.options.display.max_rows = 999
import gpflow_monitor

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.io import loadmat
from gpflow_monitor import *
print('tf_ver:', tf.__version__, 'gpflow_ver:', gpflow.__version__)
from tensorflow.python.client import device_lib
print('avail devices:\n'+'\n'.join([x.name for x in device_lib.list_local_devices()]))
from jack_utils.common import time_it
import sys
import gpflow.training.monitor as mon

# our impl
from dgp_graph import *


import argparse

parser = argparse.ArgumentParser(description='main.')
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--kernel', type=str, default='Poly1')
# parser.add_argument('--sizen', type=int)
parser.add_argument('--sizem', type=int)
parser.add_argument('--gene', type=int)
parser.add_argument('--cc', type=float)

args = parser.parse_args()

np.random.seed(123456)

loc=60
tts=200
# nodes = 100
# M = 40
# sizen = args.sizen
M = args.sizem
nodes = args.gene
cc=args.cc
fname = args.infile
inc=False
maxiter=20000

raw_data = []
gene_name = []

with open(fname) as f:
    gene_name = f.readline().strip().split()[1:]
    for l in f:
        row = [float(x) for x in l.strip().split()]
        if len(row) > 0:
            raw_data.append(row[1:])
            
data = np.asarray(raw_data)[:201]

trX1=data[:-1, :]
trY1=data[1:, :]

trX1=trX1[:,:,None]
trY1=trY1[:,:,None]

trX=trX1
trYY=trY=trY1

Z = np.stack([kmeans2(trX[:,i], M, minit='points')[0] for i in range(nodes)],axis=1)  # (M=s2=10, n, d_in=5)

adj = np.ones((nodes,nodes))# - np.eye(nodes)
adj = adj.astype('float64')
input_adj = adj # adj  / np.identity(adj.shape[0]) /  np.ones_like(adj)

time_vec = np.arange(trX.shape[0])

with gpflow.defer_build():
    m_dgpg = DGPG(trX, trYY, Z, time_vec, [1], Gaussian(), input_adj,
                  agg_op_name='concat3d', ARD=True,
                  is_Z_forward=True, mean_trainable=False, out_mf0=True,
                  num_samples=1, minibatch_size=1,
                  #kern_type='Matern32', 
                  #kern_type='RBF', 
                  kern_type=args.kernel, 
                  #wfunc='logi'
                  #kern_type='Poly1', 
                  wfunc='krbf'
                 )
    # m_sgp = SVGP(X, Y, kernels, Gaussian(), Z=Z, minibatch_size=minibatch_size, whiten=False)
m_dgpg.compile()
model = m_dgpg

session = m_dgpg.enquire_session()
optimiser = gpflow.train.AdamOptimizer(0.01)
# optimiser = gpflow.train.ScipyOptimizer()
global_step = mon.create_global_step(session)

Zcp = model.layers[0].feature.Z.value.copy()

def rmse(v1, v2):
    return np.sqrt(np.mean((v1.reshape(-1)-v2.reshape(-1))**2))

def row_norm0(ls_vec):
    #res = ls_vec / sum(ls_vec)
    #res = ls_vec**0.5 / sum(ls_vec**0.5)
    res = ls_vec
    return res

def row_norm(lss):
    tmp = np.zeros(lss.shape)
    for i in range(lss.shape[0]):
        tmp[i] = row_norm0(lss[i])
    return tmp

def get_thresh(lss, sparsity=0.1):
    tmp = row_norm(lss)
    tmp_srt = np.sort(tmp.reshape(-1))[::-1]
    idx = int(lss.shape[0]*lss.shape[1]*sparsity)+1
    th = tmp_srt[idx]
    return th

def ext_rel(ls_vec, th):
    norm_ls = row_norm0(ls_vec)
    #print(norm_ls)
    res = np.where(norm_ls > th)
    return res[0]

def print_connect(lss, th=0.6):
    for j in range(nodes):
        i = ext_rel(lss[j], th)
        print(str(i) + ' -> ' + str(j))

model.X.update_cur_n(0,cc=cc,loc=loc)
model.Y.update_cur_n(0,cc=cc,loc=loc)

pred_res, var_res = [], []

exp_path="./exp/tmp-cc%d" % int(cc)
#exp_path="./exp/temp"

print_task = mon.PrintTimingsTask()\
    .with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))\

checkpoint_task = mon.CheckpointTask(checkpoint_dir=exp_path)\
        .with_name('checkpoint')\
        .with_condition(mon.PeriodicIterationCondition(15))\

nw = np.zeros((trX.shape[0]-1, nodes, nodes))
nw[0, :, :] = model.layers[0].kern.lengthscales.value

for cur_n in range(0, trX.shape[0]-2):
# for cur_n in range(1, trX.shape[0]):
    model.X.update_cur_n(cur_n,cc=cc,loc=loc)
    model.Y.update_cur_n(cur_n,cc=cc,loc=loc)
    with mon.LogdirWriter(exp_path) as writer:
        tensorboard_task = mon.ModelToTensorBoardTask(writer, model)\
            .with_name('tensorboard')\
            .with_condition(mon.PeriodicIterationCondition(100))\
            .with_exit_condition(True)
        monitor_tasks = [] # [print_task, tensorboard_task]

        with mon.Monitor(monitor_tasks, session) as monitor:
            #optimiser.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)
            model.layers[0].feature.Z.assign(Zcp.copy())
            model.layers[0].kern.lengthscales.assign(np.ones((nodes, nodes)))
            optimiser.minimize(model, step_callback=monitor, maxiter=maxiter)
            nw[cur_n, :, :] = model.layers[0].kern.lengthscales.value
            
    teX = trX[cur_n+1].reshape(1, nodes)

    S=100
    m, v = model.predict_y(teX, S)
    pred = np.mean(m, axis=0)
    var = np.mean(v, axis=0)
    if inc:
        pred += teX

    pred_res.append(pred)
    var_res.append(var)
    print('STEP %d - loss: %f' % (cur_n, rmse(pred, trY[cur_n+2])))
#     break

pred_mat = np.asarray(pred_res).squeeze()
var_mat = np.asarray(var_res).squeeze()
with open(args.outfile, 'wb') as fout:
    np.savez(fout, nw=nw, pred=pred_mat, var=var_mat)
