import os.path as osp
from dataset import custom_dataset
import numpy as np
from graph_utils import edge_index_to_adj_matrix

from reorder import ReorderStrategy


def show_dataset(dataset):
    print(edge_index_to_adj_matrix(
        dataset.edge_index, dataset.num_nodes))


path = osp.join(
    "/home/xiaosiyier/projects/OSDI21_AE/my-test-graphs/", "test_rabbit.txt")
dataset = custom_dataset(path, 16, 10, load_from_txt=True, verbose=True)
show_dataset(dataset)
dataset.reorder(ReorderStrategy.RABBIT)
show_dataset(dataset)
dataset.reorder(ReorderStrategy.METIS)
show_dataset(dataset)
