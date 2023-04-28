import numpy as np
import pymetis
import torch
from graph_utils import *
from random import random
from math import floor


def metis_reorder(edge_index, n):
    '''
    edge_index(np.ndarray)
    return: iperm
    '''
    adj_list = edge_index_to_adj_list(edge_index, n)
    print('to node_nd')
    print(adj_list)
    node_nd = pymetis.nested_dissection(adjacency=adj_list)
    iperm = node_nd[1]
    return iperm


def rand(n): return floor(random()*n)


def test_gen_edge_index(n, density):
    edges = floor(n*n*density)//2
    src = []
    dst = []
    set = []
    for i in range(edges):
        s = rand(n)
        d = rand(n)
        if [s, d] not in set and s != d:
            set.append([s, d])
            set.append([d, s])
            src.append(s)
            dst.append(d)
            src.append(d)
            dst.append(s)
    return [src, dst]


if __name__ == "__main__":
    n = 6
    edge_index = test_gen_edge_index(n, 0.5)
    print(edge_index_to_adj_matrix(edge_index, n))
    print()
    edge_index = metis_reorder(edge_index, n)
