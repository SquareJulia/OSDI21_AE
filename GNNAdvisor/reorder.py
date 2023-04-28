from enum import Enum
import rabbit
from numpy import random
import torch
from driver_pymetis import metis_reorder


class ReorderStrategy(Enum):
    NONE = 0
    RANDOM = 1
    DEGREE = 2
    RABBIT = 3
    METIS = 4


def reorder_with_strategy(reorder_strategy, edge_index, degrees, num_nodes):
    ''' Reorder vertices with the specific strategy. Modify edge_index in-place.
    '''
    new_edge_index = edge_index
    if reorder_strategy is ReorderStrategy.RANDOM:
        reorder_random(edge_index, num_nodes)
    elif reorder_strategy is ReorderStrategy.DEGREE:
        reorder_degree(edge_index, degrees)
    elif reorder_strategy is ReorderStrategy.RABBIT:
        new_edge_index = rabbit.reorder(
            torch.IntTensor(edge_index)).tolist()
    elif reorder_strategy is ReorderStrategy.METIS:
        reorder_metis(edge_index, num_nodes)
    return new_edge_index


def map_edge_index(iperm, edge_index):
    ''' iperm is the new permutation of the nodes.
        Modify edge_index in-place.'''
    for e in range(len(edge_index[0])):
        edge_index[0][e] = iperm[edge_index[0][e]]
        edge_index[1][e] = iperm[edge_index[1][e]]


def reorder_random(edge_index, num_nodes):
    ''' Reorder vertices by randomly. Modify edge_index in-place.
    '''
    iperm = random.permutation(int(num_nodes))
    map_edge_index(iperm, edge_index)


def reorder_degree(edge_index, degrees, desc=False):
    ''' Reorder vertices by degrees. Modify edge_index in-place.
    '''
    n = len(degrees)
    order = [[i, degrees[i]] for i in range(n)]
    order.sort(key=lambda v: v[1], reverse=desc)
    mapping = [0]*n  # old id => new id
    for v in range(n):
        mapping[order[v][0]] = v
    map_edge_index(mapping, edge_index)


def reorder_metis(edge_index, num_nodes):
    ''' Reorder vertices with METIS. Modify edge_index in-place.
    '''
    iperm = metis_reorder(edge_index, num_nodes)
    print('Got iperm')
    map_edge_index(iperm, edge_index)
