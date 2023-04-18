
import torch
import numpy as np
from scipy.sparse import coo_matrix


def adj_list_asc_neighbours(adj_list):
    ''' Sort the neighbours of each source node by id ascending.
    '''
    for adj in adj_list:
        adj.sort()
    return adj_list


def edge_index_to_adj_list(edge_index, n):
    ''' Convert edge_index(np.ndarray/list/Tensor of shape (2 x num_edges)) to 
        adjacency list(list[list[]])
        Note: the neighbours of each source node are sorted by id ascending.
        n: number of nodes
    '''
    adj_list = [[] for i in range(n)]
    if torch.is_tensor(edge_index):
        for i in range(edge_index.size(1)):
            src = edge_index[0][i]
            dst = edge_index[1][i]
            adj_list[src].append(dst.item())
    else:
        for i in range(len(edge_index[0])):
            src = edge_index[0][i]
            dst = edge_index[1][i]
            adj_list[src].append(dst)

    return adj_list_asc_neighbours(adj_list)


def edge_index_to_adj_matrix(edge_index, num_nodes):
    ''' Convert edge_index(list/IntTensor) to adjacency matrix.
    '''
    val = [1] * len(edge_index[0])
    scipy_coo = coo_matrix((val, edge_index),
                           shape=(num_nodes, num_nodes))
    return scipy_coo.toarray().astype('float32')
