import numpy as np
import pymetis
import torch
from utils import *
from random import random
from math import floor

def metis_reorder(in_edge_index, barrier,n):
    '''
    in_edge_index(np.ndarray/list/Tensor)
    barrier(int)

    return: edge_index(torch.IntTensor)
    '''
    adj_list = edge_index_to_adj_list(in_edge_index,n)
    node_nd = pymetis.nested_dissection(adjacency=adj_list)
    iperm = node_nd[1]

    num_edges = in_edge_index.size(1) if torch.is_tensor(in_edge_index) else len(in_edge_index[0])

    out_edge_index = torch.zeros(2,num_edges)
    for e in range(num_edges):
        src = in_edge_index[0][e]
        dst = in_edge_index[1][e]
        out_edge_index[0][e] = iperm[src] if src < barrier else src
        out_edge_index[1][e] = iperm[dst] if dst < barrier else dst
    return out_edge_index

rand=lambda n: floor(random()*n)
def test_gen_edge_index(n,density):
    edges=floor(n*n*density)//2
    src=[]
    dst=[]
    set=[]
    for i in range(edges):
        s=rand(n)
        d=rand(n)
        if [s,d] not in set and s!=d:
            set.append([s,d])
            set.append([d,s])
            src.append(s)
            dst.append(d)
            src.append(d)
            dst.append(s)
    return [src,dst]


if __name__ == "__main__":
    n=6
    edge_index=test_gen_edge_index(n, 0.5)
    print(edge_index_to_adj_matrix(edge_index, n))
    print()
    edge_index=metis_reorder(edge_index,n,n)
    print(edge_index_to_adj_matrix(edge_index, n))
    