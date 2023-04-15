from itertools import accumulate
from functools import reduce
import torch
from scipy.sparse import *
import numpy as np


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


def split_adj_list_by_density(adj_list, density, TILE_ROW, TILE_COL, num_nodes):
    ''' Split adj_list (list[list[]]) into 2 adj_lists by the density
        of each tile (of size TILE_ROW*TILE_COL) in the corresponding adjacency matrix.
        Note: The last row%TILE_ROW rows belong to dense_adj without splitting.
            (to make the row dim divisible by TILE_ROW for SparseRT)
        Return: dense_adj(list[list[]])
                sparse_adj_transposed(list[list[]])
    '''
    dense_adj = [[] for i in range(num_nodes)]
    assert num_nodes > TILE_ROW  # TODO
    sparse_adj = [[] for i in range(num_nodes-(num_nodes % TILE_ROW))]
    i = 0
    while i+TILE_ROW <= num_nodes:
        col_start = [0]*TILE_ROW
        col_end = [-1]*TILE_ROW
        tile_col_start = 0
        while tile_col_start < num_nodes:
            nnz = 0
            for row in range(TILE_ROW):
                while col_end[row] < len(adj_list[i+row])-1 and adj_list[i+row][col_end[row]+1] < tile_col_start+TILE_COL:
                    col_end[row] += 1
                nnz += col_end[row]-col_start[row]+1
            target_adj = sparse_adj if nnz / \
                (TILE_ROW*min(TILE_COL, num_nodes-tile_col_start)) < density else dense_adj
            for row in range(TILE_ROW):
                if col_end[row] >= col_start[row]:
                    target_adj[row+i] += adj_list[row +
                                                  i][col_start[row]:col_end[row]+1]
                    col_start[row] = col_end[row]+1
            tile_col_start += TILE_COL
        i += TILE_ROW
    while i < num_nodes:
        dense_adj[i] = adj_list[i][:]
        i += 1
    return dense_adj, sparse_adj


def adj_list_to_csr(adj_list):
    ''' Convert adj_list (list[list[]]) to row_pointers and column_index
    Return: row_pointers,column_index (torch.IntTensor)
    '''
    n = len(adj_list)
    degrees = [len(x) for x in adj_list]
    row_pointers = [0, *accumulate(degrees)]
    column_index = reduce(lambda x, y: x+y, adj_list)
    return torch.IntTensor(row_pointers), torch.IntTensor(column_index)


def reorder_by_degree(edge_index, degrees, desc=True):
    ''' Reorder vertices by degrees. Modify edge_index in-place.
    '''
    n = len(degrees)
    order = [[i, degrees[i]] for i in range(n)]
    order.sort(key=lambda v: v[1], reverse=desc)
    mapping = [0]*n  # old id => new id
    for v in range(n):
        mapping[order[v][0]] = v
    for e in range(len(edge_index[0])):
        edge_index[0][e] = mapping[edge_index[0][e]]
        edge_index[1][e] = mapping[edge_index[1][e]]


def degrees_from_edge_index(edge_index, num_nodes):
    ''' Generate degrees array from edge index. Assume the graph is undirected.
    '''
    # print('num_nodes:', num_nodes)  # 3327
    # degrees = [0]*num_nodes
    degrees = np.zeros(num_nodes, dtype=np.float32)
    # print('len(degrees):', len(degrees))  # 1
    for src in edge_index[0]:
        if src >= len(degrees):
            print(edge_index[0], edge_index[1])
            print(len(degrees))
        degrees[src] += 1
    return degrees


def edge_index_to_adj_matrix(edge_index, num_nodes):
    ''' Convert edge_index(list/IntTensor) to adjacency matrix.
    '''
    val = [1] * len(edge_index[0])
    scipy_coo = coo_matrix((val, edge_index),
                           shape=(num_nodes, num_nodes))
    return scipy_coo.toarray().astype('float32')


def adj_list_to_coo_matrix(adj_list, num_nodes):
    rows = []
    for vid, adj in enumerate(adj_list):
        rows.extend([vid]*len(adj))
    vals = [1]*len(rows)
    cols = reduce(lambda x, y: x+y, adj_list)
    return coo_matrix((vals, (rows, cols)),
                      shape=(len(adj_list), num_nodes), dtype=np.float32)


def adj_list_to_adj_matrix(adj_list, num_nodes):
    print(adj_list)
    rows = []
    for vid, adj in enumerate(adj_list):
        rows.extend([vid]*len(adj))
    vals = [1]*len(rows)
    cols = reduce(lambda x, y: x+y, adj_list)
    scipy_coo = coo_matrix((vals, (rows, cols)),
                           shape=(len(adj_list), num_nodes))
    return scipy_coo.toarray().astype('float32')
