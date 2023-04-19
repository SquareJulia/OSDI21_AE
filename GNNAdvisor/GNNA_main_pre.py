import sys
import time
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from tqdm import *
from scipy.sparse import *

import GNNAdvisor as GNNA           # import GNNAdvisor

from gnn_conv import *
from dataset import *

import os
from cuda import cuda, nvrtc
from utils import *
import ctypes
from sparsert import SparseRTLayer
import log
from constants import *


parser = argparse.ArgumentParser()
# Dataset related parameters.
parser.add_argument("--dataDir", type=str,
                    default="../osdi-ae-graphs", help="the path to graphs")
parser.add_argument("--dataset", type=str,
                    default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96,
                    help="input embedding dimension size")
parser.add_argument("--hidden", type=int, default=16,
                    help="hidden dimension size")
parser.add_argument("--classes", type=int, default=22,
                    help="output classes size")


# Manually set the performance related parameters
# Mode-divergence related
parser.add_argument("--density", type=float, default=0.001,
                    help="Density threshold for splitting the adjacency matrix into dense and sparse tiles. Default:0.001")
parser.add_argument("--A_tileDim", type=int, default=80,
                    help="Dim of each subtile of A in the spliited adjacency matrix AB. Default:80")
parser.add_argument("--B_tileDim", type=int, default=80,
                    help="Dim of each subtile of B in the spliited adjacency matrix AB. Default:80")
# SparseRT related
parser.add_argument("--A_blockDim", type=int, default=8,
                    help="(SparseRT) Per-block workload of dims of A in the spliited adjacency matrix AB. Default:8")
parser.add_argument("--Gy_input", type=int, default=1,
                    help="(SparseRT) Number of thread groups for each A_block, input-layer. Default:1")
parser.add_argument("--Gy_hidden", type=int, default=1,
                    help="(SparseRT) Number of thread groups for each A_block, hidden-layer. Default:1")
parser.add_argument("--C_blocks_input", type=int, default=1,
                    help="(SparseRT) Number of thread blocks to deal with the input-layer C_dim. Default:1")
parser.add_argument("--C_blocks_hidden", type=int, default=1,
                    help="(SparseRT) Number of thread blocks to deal with the hidden-layer C_dim. Default:1")
# GNNAdvisor related
parser.add_argument("--partSize", type=int, default=32,
                    help="neighbor-group size")
parser.add_argument("--dimWorker", type=int, default=16,
                    help="number of worker threads (MUST < 32)")
parser.add_argument("--warpPerBlock", type=int, default=8,
                    help="number of warp per block, recommended: GCN: 8, GIN: 2")
parser.add_argument("--sharedMem", type=int, default=100,
                    help="shared memory size of each block (Quadro P6000 64(KB) sm_61), default=100(KB) for RTX3090 sm_86")


# Additional flags for studies.
parser.add_argument('--manual_mode', type=str, choices=[
                    'True', 'False'], default='True', help="True: use manual config, False: auto config, default: True")
parser.add_argument('--verbose_mode', type=str, choices=[
                    'True', 'False'], default='False', help="True: verbose mode, False: simple mode, default: False")

parser.add_argument('--loadFromTxt', type=str, choices=['True', 'False'], default='False',
                    help="True: load the graph TXT edge list, False: load from .npy, default: False (load from npz fast)")
parser.add_argument('--single_spmm', type=str, choices=['True', 'False'], default='False',
                    help="True: profile the single SpMM (neighbor aggregation) kernel for number epoches times")
parser.add_argument('--verify_spmm', type=str, choices=['True', 'False'], default='False',
                    help="True: verify the output correctness of a single SpMM (neighbor aggregation) kernel against the CPU reference implementation.")

parser.add_argument('--reorder_strategy', type=str, choices=['None', 'random', 'degree', 'rabbit', 'METIS'], default='None',
                    help="Strategy of reordering the input graph. Possible choices are: None, random, degree, rabbit, METIS; Default: None.")

args = parser.parse_args()
print(args)

partSize, dimWorker, warpPerBlock, sharedMem = args.partSize, args.dimWorker, args.warpPerBlock, args.sharedMem
A_blockDim, Gy_input, Gy_hidden, C_blocks_input, C_blocks_hidden = args.A_blockDim, args.Gy_input, args.Gy_hidden, args.C_blocks_input, args.C_blocks_hidden
density, A_tileDim, B_tileDim = args.density, args.A_tileDim, args.B_tileDim
manual_mode = args.manual_mode == 'True'
verbose_mode = args.verbose_mode == 'True'

reorder_strategy_name = reorder_strategy.upper()
loadFromTxt = args.loadFromTxt == 'True'
single_spmm = args.single_spmm == 'True'
verify_spmm = args.verify_spmm == 'True'

# requires GPU for evaluation.
assert torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_printoptions(precision=2)
####################################
# loading data from files
####################################
if loadFromTxt:
    path = osp.join(args.dataDir, args.dataset)
    dataset = custom_dataset(path, args.dim, args.classes,
                             load_from_txt=True, verbose=verbose_mode)
else:
    path = osp.join(args.dataDir, args.dataset+".npz")
    dataset = custom_dataset(path, args.dim, args.classes,
                             load_from_txt=False, verbose=verbose_mode)


####################################
# Building input property profile.
####################################
inputInfo = inputProperty(args.hidden, dataset, manual_mode, verbose_mode,
                          reorder_strategy_name,
                          density, A_tileDim, B_tileDim,
                          partSize, dimWorker, warpPerBlock, sharedMem,
                          A_blockDim, Gy_input, Gy_hidden, C_blocks_input, C_blocks_hidden)

####################################
# Decider for parameter selection.(Reorder steps included)
####################################
inputInfo.decider()
dataset.plus_identity_matrix()
dataset.edge_index_to_adj_list()
dataset.split_by_density(
    inputInfo.density, inputInfo.A_tileDim, inputInfo.B_tileDim)

inputInfo = inputInfo.set_input()
if verbose_mode:
    print('----------------------------')
    inputInfo.print_param_general()
    inputInfo.print_param_layerwise()

inputInfo = inputInfo.set_hidden()
if verbose_mode:
    inputInfo.print_param_layerwise()


####################################
# SparseRT
####################################

degrees_file, AB_file = dataset.save_for_sparsert()
inputLayerSpRT = SparseRTLayer(degrees_file, AB_file, inputInfo,
                               inputInfo.outputDim_input, inputInfo.C_blocks_input, inputInfo.Gy_input, verbose_mode)
hiddenLayerSpRT = SparseRTLayer(degrees_file, AB_file, inputInfo,
                                inputInfo.outputDim_hidden, inputInfo.C_blocks_hidden, inputInfo.Gy_hidden, verbose_mode)


start = time.perf_counter()
inputLayerSpRT.gen_ptx_and_cubin(inputInfo)
hiddenLayerSpRT.gen_ptx_and_cubin(inputInfo)
elapsed = time.perf_counter() - start
if verbose_mode:
    log.done(
        "# SparseRT generate .ptx & .cubin, and get function handle (s): {:.3f}".format(elapsed))

####################################
# Building neighbor partitioning.
####################################
dataset.GNNA_gen_csr()

if len(dataset.dense_column_index) > 0:
    start = time.perf_counter()
    partPtr, part2Node = GNNA.build_part(
        inputInfo.partSize, torch.IntTensor(dataset.dense_row_pointers))
    elapsed = time.perf_counter() - start
    log.done("# Build nb_part (s): {:.3f}".format(elapsed))
else:
    partPtr = torch.zeros(0)
    part2Node = torch.zeros(0)

inputInfo.partPtr = partPtr.int()
inputInfo.part2Node = part2Node.int()

####################################
# Saving preprocessing results.
####################################

pre_dir = pre_dir_data_template(path)+pre_dir_params_template(inputInfo)
pre_dataset = pre_dir+PREPROCESSED_DATASET
pre_inputInfo = pre_dir+PREPROCESSED_INPUT_INFO
pre_inputLayerSpRT = pre_dir+PREPROCESSED_INPUT_LAYER_SPRT
pre_hiddenLayerSpRT = pre_dir+PREPROCESSED_HIDDEN_LAYER_SPRT

if not osp.exists(pre_dir):
    os.makedirs(pre_dir)
else:
    remove_files_if_exists(pre_dataset, pre_inputInfo,
                           pre_inputLayerSpRT, pre_hiddenLayerSpRT)
torch.save(dataset, pre_dataset)
torch.save(inputInfo, pre_inputInfo)
torch.save(inputLayerSpRT, pre_inputLayerSpRT)
torch.save(hiddenLayerSpRT, pre_hiddenLayerSpRT)

log.done('# Saved preprocessing results!')
