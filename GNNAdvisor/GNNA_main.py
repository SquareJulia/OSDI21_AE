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

# Model training related parameters.
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'gin'],  help="GCN or GIN")
parser.add_argument("--num_epoches", type=int, default=200,
                    help="number of epoches for training, default=200")

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
arser.add_argument("--partSize", type=int, default=32,
                   help="neighbor-group size")
parser.add_argument("--dimWorker", type=int, default=32,
                    help="number of worker threads (MUST < 32)")
parser.add_argument("--warpPerBlock", type=int, default=4,
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

parser.add_argument('--enable_sort_by_degree', type=str, choices=['True', 'False'], default='False',
                    help="True: enable reordering by degrees, False, disable reordering by degrees, default: False (disable for both manual and auto mode).")
parser.add_argument('--enable_rabbit', type=str, choices=['True', 'False'], default='False',
                    help="True: enable rabbit reordering, False, disable rabbit reordering, default: False (disable for both manual and auto mode).")
parser.add_argument("--rabbitRatio", type=float, default=0.8,
                    help="Ratio of (possibly reordered by degree descending) vertices to be reordered by rabbit, default=0.8")


args = parser.parse_args()
print(args)

partSize, dimWorker, warpPerBlock, sharedMem = args.partSize, args.dimWorker, args.warpPerBlock, args.sharedMem
A_blockDim, Gy_input, Gy_hidden, C_blocks_input, C_blocks_hidden = args.A_blockDim, args.Gy_input, args.Gy_hidden, args.C_blocks_input, args.C_blocks_hidden
rabbitRatio = args.rabbitRatio
density, A_tileDim, B_tileDim = args.density, args.A_tileDim, args.B_tileDim
manual_mode = args.manual_mode == 'True'
verbose_mode = args.verbose_mode == 'True'
enable_rabbit = args.enable_rabbit == 'True'
enable_sort_by_degree = args.enable_sort_by_degree == 'True'
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
inputInfo = inputProperty(args.hidden,dataset,manual_mode,verbose_mode,
                enable_rabbit,enable_sort_by_degree,rabbitRatio,
                partSize,dimWorker,warpPerBlock,sharedMem,
                A_blockDim,Gy_input,Gy_hidden,C_blocks_input,C_blocks_hidden)

####################################
# Decider for parameter selection.
####################################
inputInfo.decider()

inputInfo = inputInfo.set_input()
if verbose_mode:
    print('----------------------------')
    inputInfo.print_param()
    print()

inputInfo = inputInfo.set_hidden()
if verbose_mode:
    inputInfo.print_param()
    print()
    inputInfo.print_param_SpRT()
    print()
    print('----------------------------')

num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index = dataset.column_index
row_pointers = dataset.row_pointers
degrees = dataset.degrees
# if verbose_mode:
#     print('degrees:')
#     print(degrees)
#     print('column_index:')
#     print(column_index)
#     print('row_pointers:')
#     print(row_pointers)
# sys.exit(0)

####################################
# SparseRT
####################################

BA_npy = dataset.save_transposed_sparse_matrix_npy(
    inputInfo.modeBarrier) if inputInfo.modeBarrier > 0 else ''

inputLayerSpRT = SparseRTLayer(BA_npy, inputInfo,
                               inputInfo.outputDim_input, inputInfo.C_blocks_input, inputInfo.Gy_input, verbose_mode)
hiddenLayerSpRT = SparseRTLayer(BA_npy, inputInfo,
                                inputInfo.outputDim_hidden, inputInfo.C_blocks_hidden, inputInfo.Gy_hidden, verbose_mode)

if inputInfo.modeBarrier > 0:
    start = time.perf_counter()
    inputLayerSpRT.gen_ptx_and_cubin()
    inputLayerSpRT.get_func_handle()
    hiddenLayerSpRT.gen_ptx_and_cubin()
    hiddenLayerSpRT.get_func_handle()
    elapsed = time.perf_counter() - start
    if verbose_mode:
        log.done(
            "# SparseRT generate .ptx & .cubin, and get function handle (s): {:.3f}".format(elapsed))

####################################
# Building neighbor partitioning.
####################################


if inputInfo.modeBarrier < num_nodes:
    start = time.perf_counter()
    partPtr, part2Node = GNNA.build_part(
        inputInfo.partSize, dataset.row_pointers, inputInfo.modeBarrier)
    elapsed = time.perf_counter() - start
    if verbose_mode:
        log.done("# Build nb_part (s): {:.3f}".format(elapsed))
else:
    partPtr = torch.zeros(0)
    part2Node = torch.zeros(0)

dataset.row_pointers = dataset.row_pointers.to(device)
dataset.column_index = dataset.column_index.to(device)
inputInfo.partPtr = partPtr.int().to(device)
inputInfo.part2Node = part2Node.int().to(device)

####################################
# Verifing a single SpMM kernel
# against the CPU reference.
####################################
if verify_spmm:
    from unitest import *
    valid = Verification(args.hidden,
                         dataset.row_pointers, dataset.column_index, degrees,
                         inputInfo.partPtr, inputInfo.part2Node,
                         partSize, dimWorker, warpPerBlock,
                         inputLayerSpRT.cu_function, inputLayerSpRT.A_blocks, inputLayerSpRT.C_blocks, inputLayerSpRT.Block_size, inputLayerSpRT.ctx)
    valid.compute()
    valid.reference(dataset.edge_index, dataset.a_times_d(
        inputInfo.modeBarrier), dataset.num_nodes)
    valid.compare()
    sys.exit(0)


####################################
# Profiling a single SpMM kernel
####################################
if single_spmm:
    from unitest import *
    valid = Verification(args.hidden,
                         inputInfo.row_pointers, dataset.column_index, degrees,
                         inputInfo.partPtr, inputInfo.part2Node,
                         partSize, dimWorker, warpPerBlock)
    valid.profile_spmm(round=args.num_epoches)
    sys.exit(0)

####################################
# Building GNN model
####################################
# if args.model == 'gcn':

a_hat_hat_for_test = torch.FloatTensor(dataset.a_times_d())


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden,
                             inputLayerSpRT, a_hat_hat_for_test)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes,
                             hiddenLayerSpRT, a_hat_hat_for_test)

    def forward(self):
        x = dataset.x
        x = F.relu(self.conv1(x, inputInfo.set_input()))
        x = self.conv2(x, inputInfo.set_hidden())
        return F.log_softmax(x, dim=1)


model, dataset = Net().to(device), dataset.to(device)
if verbose_mode:
    print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

####################################
# Define training function.
####################################


def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[:], dataset.y[:])
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    # dry run
    # for _ in range(10):
    #     train()
    # exit(0)

    torch.cuda.synchronize()
    start_train = time.perf_counter()
    # for _ in tqdm(range(1, args.num_epoches + 1)):
    #     train()
    train()
    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_train

    print('Time (ms): {:.3f}'.format(train_time*1e3/args.num_epoches))
    print()
