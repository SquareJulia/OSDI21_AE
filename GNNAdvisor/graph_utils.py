from itertools import accumulate
from functools import reduce
import torch


def edge_index_to_adj_list(edge_index, n):
    ''' Convert edge_index(np.ndarray/list/Tensor of shape (2 x num_edges)) to 
    list[list[]]
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
    return adj_list


def split_adj_list_by_density(adj_list, density, TILE_ROW, TILE_COL, num_nodes):
    ''' Split adj_list (list[list[]]) into 2 adj_lists by the density
        of each tile (of size TILE_ROW*TILE_COL) in the corresponding adjacency matrix.
        Note: The last row%TILE_ROW rows belongs to dense_adj without splitting.
            (to make the row dim divisible by TILE_ROW for SparseRT)
        Return: dense_adj,sparse_adj (list[list[]])
    '''
    dense_adj = [[] for i in range(num_nodes)]
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
    Return: row_pointers,column_index (list)
    '''
    n = len(adj_list)
    degrees = [len(x) for x in adj_list]
    row_pointers = [0, *accumulate(degrees)]
    column_index = reduce(lambda x, y: x+y, adj)
    return row_pointers, column_index


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
    degrees = [0]*num_nodes
    for src in edge_index[0]:
        degrees[src] += 1
    return degrees
