
import torch
import GNNAdvisor as GNNA           # import GNNAdvisor
from dataset import custom_dataset
import time
from graph_utils import *
import os.path as osp
import log

modeBarrier = 4
partSize = 2
path = osp.join(
    "/home/xiaosiyier/projects/OSDI21_AE/my-test-graphs/", "g6nodes.txt")
dataset = custom_dataset(path, 16, 10, load_from_txt=True, verbose=True)
# reorder
dataset.reorder_rabbit_flag = True
dataset.reorder_by_degree_flag = True
dataset.print_adj_matrix()
dataset.degree_reorder()
dataset.print_adj_matrix()
dataset.rabbit_reorder(modeBarrier)
dataset.print_adj_matrix()
dataset.edge_index_to_adj_list()
# split
dataset.split_by_density(2, 2, 2)
# GNNA
row_pointers, column_index = adj_list_to_csr(dataset.dense_adj)
print('row_pointers:')
print(row_pointers)
print('column_index:')
print(column_index)
if len(column_index) > 0:
    start = time.perf_counter()
    partPtr, part2Node = GNNA.build_part(
        partSize, row_pointers)
    elapsed = time.perf_counter() - start
    log.done("# Build nb_part (s): {:.3f}".format(elapsed))
else:
    partPtr = torch.zeros(0)
    part2Node = torch.zeros(0)
print('partPtr')
print(partPtr)
print('part2Node')
print(part2Node)

# dataset.row_pointers = dataset.row_pointers.to(device)
# dataset.column_index = dataset.column_index.to(device)
# inputInfo.partPtr = partPtr.int().to(device)
# inputInfo.part2Node = part2Node.int().to(device)
