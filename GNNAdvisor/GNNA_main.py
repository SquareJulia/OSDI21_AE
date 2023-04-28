import log
from sparsert import SparseRTLayer
import ctypes
from utils import *
from cuda import cuda, nvrtc
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
from constants import *

from reorder import ReorderStrategy


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


# Additional flags for studies.
parser.add_argument('--verbose_mode', type=str, choices=[
                    'True', 'False'], default='False', help="True: verbose mode, False: simple mode, default: False")
parser.add_argument('--loadFromTxt', type=str, choices=['True', 'False'], default='False',
                    help="True: load the graph TXT edge list, False: load from .npy, default: False (load from npz fast)")

parser.add_argument('--reorder_strategy', type=str, choices=['None', 'random', 'degree', 'rabbit', 'METIS'], default='None',
                    help="Strategy of reordering the input graph. Possible choices are: None, random, degree, rabbit, METIS; Default: None.")


# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)
loadFromTxt = args.loadFromTxt == 'True'
dim, hidden, classes = args.dim, args.hidden, args.classes
density, A_tileDim, B_tileDim = args.density, args.A_tileDim, args.B_tileDim
verbose_mode = args.verbose_mode == 'True'

reorder_strategy_name = args.reorder_strategy.upper()
if reorder_strategy_name not in ReorderStrategy.__members__:
    raise ValueError('Reorder strategy not valid !!!')
reorder_strategy = ReorderStrategy[reorder_strategy_name]


# requires GPU for evaluation.
assert torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_printoptions(precision=2)

####################################
# Restoring preprocessing results.
####################################

path = osp.join(
    args.dataDir, args.dataset if loadFromTxt else args.dataset+'.npz')
pre_dir_data = pre_dir_data_template(path)
pre_dir_params = pre_dir_params_template_base(
    dim, hidden, classes, reorder_strategy, density, A_tileDim)
pre_dir = pre_dir_data+pre_dir_params
pre_dataset = pre_dir+PREPROCESSED_DATASET
pre_inputInfo = pre_dir+PREPROCESSED_INPUT_INFO
pre_inputLayerSpRT = pre_dir+PREPROCESSED_INPUT_LAYER_SPRT
pre_hiddenLayerSpRT = pre_dir+PREPROCESSED_HIDDEN_LAYER_SPRT
pre_SparseRT_dir = pre_dir+PREPROCESSED_SPARSERT_DIR
if not osp.exists(pre_dir) or not all_files(pre_dataset, pre_inputInfo, pre_inputLayerSpRT, pre_hiddenLayerSpRT) or not osp.isdir(pre_SparseRT_dir):
    raise FileNotFoundError(
        'Preprocessing files not found! Run "GNNA_main_pre.py first!')

dataset = torch.load(pre_dataset)
inputInfo = torch.load(pre_inputInfo)
inputLayerSpRT = torch.load(pre_inputLayerSpRT)
hiddenLayerSpRT = torch.load(pre_hiddenLayerSpRT)

log.done('# Loaded preprocessing results!')

####################################
# Preparing data for training and testing.
####################################

inputInfo.dataset_obj = dataset  # IMPORTANT

torch.cuda.set_device(0)
dataset.x = dataset.x.cuda()
dataset.y = dataset.y.cuda()
dataset.dense_row_pointers = dataset.dense_row_pointers.cuda()
dataset.dense_column_index = dataset.dense_column_index.cuda()
inputInfo.partPtr = inputInfo.partPtr.cuda()
inputInfo.part2Node = inputInfo.part2Node.cuda()
dataset.degrees_gpu = dataset.degrees_cpu.cuda()


inputLayerSpRT.get_ctx()
inputLayerSpRT.get_func_handle()
hiddenLayerSpRT.get_ctx()
hiddenLayerSpRT.get_func_handle()


a_hat_hat_for_test = torch.FloatTensor(dataset.calc_d_a_d())


####################################
# Define model.
####################################


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


model, dataset = Net().cuda(), dataset.cuda()
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
    for _ in range(10):
        train()
    # exit(0)

    torch.cuda.synchronize()
    start_train = time.perf_counter()
    for _ in tqdm(range(1, args.num_epoches + 1)):
        train()
    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_train

    print('Time (ms): {:.3f}'.format(
        train_time*1e3/args.num_epoches))
    print()
