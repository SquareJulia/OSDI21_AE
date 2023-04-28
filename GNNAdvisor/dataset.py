#!/usr/bin/env python3
import torch
import numpy as np
import time
import dgl
import os.path as osp
import os

from scipy.sparse import *
from constants import *
import log
from graph_utils import *
from reorder import *


def func(x):
    '''
    node degrees function
    '''
    if x > 0:
        return x
    else:
        return 1


class custom_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """

    def __init__(self, path, dim, num_class, load_from_txt=True, verbose=False):
        super(custom_dataset, self).__init__()

        self.nodes = set()

        self.load_from_txt = load_from_txt
        self.path = path

        self.num_nodes = 0
        self.num_features = dim
        self.num_classes = num_class
        self.edge_index = None

        self.verbose_flag = verbose

        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.init_edges(path)
        self.init_embedding(dim)
        self.init_labels(num_class)

        train = 1
        val = 0.3
        test = 0.1
        self.train_mask = [1] * int(self.num_nodes * train) + \
            [0] * (self.num_nodes - int(self.num_nodes * train))
        self.val_mask = [1] * int(self.num_nodes * val) + \
            [0] * (self.num_nodes - int(self.num_nodes * val))
        self.test_mask = [1] * int(self.num_nodes * test) + \
            [0] * (self.num_nodes - int(self.num_nodes * test))
        self.train_mask = torch.BoolTensor(self.train_mask).cuda()
        self.val_mask = torch.BoolTensor(self.val_mask).cuda()
        self.test_mask = torch.BoolTensor(self.test_mask).cuda()

    def init_edges(self, path):
        self.g = dgl.DGLGraph()

        # loading from a txt graph file
        if self.load_from_txt:
            fp = open(path, "r")
            src_li = []
            dst_li = []
            start = time.perf_counter()
            for line in fp:
                src, dst = line.strip('\n').split()
                src, dst = int(src), int(dst)
                src_li.append(src)
                dst_li.append(dst)
                self.nodes.add(src)
                self.nodes.add(dst)

            self.num_nodes = max(self.nodes) + 1  # buggy

            dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (txt) {:.3f}s ".format(dur))

        # loading from a .npz graph file
        else:
            if not path.endswith('.npz'):
                raise ValueError("graph file must be a .npz file")

            start = time.perf_counter()
            graph_obj = np.load(path)
            src_li = graph_obj['src_li']
            dst_li = graph_obj['dst_li']

            self.num_nodes = graph_obj['num_nodes']
            dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (npz)(s): {:.3f}".format(dur))

        self.num_edges = len(src_li)
        self.avg_degree = self.num_edges / self.num_nodes
        self.avg_edgeSpan = np.mean(np.abs(np.subtract(src_li, dst_li)))

        if self.verbose_flag:
            print('# nodes: {}'.format(self.num_nodes))
            print("# avg_degree: {:.2f}".format(self.avg_degree))
            print("# avg_edgeSpan: {}".format(int(self.avg_edgeSpan)))

        self.edge_index = np.stack([src_li, dst_li])
        if self.verbose_flag:
            log.info('# edges: {}'.format(self.num_edges))
        self.avg_density = self.num_edges/(self.num_nodes*self.num_nodes)
        if self.verbose_flag:
            log.info('average density: {:.3f}'.format(self.avg_density))
        self.degrees_cpu = degrees_from_edge_index(
            self.edge_index, self.num_nodes)
        self.degrees_gpu = None

        # self.val = [1] * self.num_edges
        # self.generate_csr_and_degrees()

    def plus_identity_matrix(self):
        ''' Operate on edge_index. Modify edge_index, num_edges
        '''
        self.edge_index = [np.concatenate((self.edge_index[0], [i for i in range(self.num_nodes)])),
                           np.concatenate((self.edge_index[1], [i for i in range(self.num_nodes)]))]
        self.num_edges += self.num_nodes

    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, dim)

    def init_labels(self, num_class):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.ones(self.num_nodes).long()

    def reorder(self, reorder_strategy):
        if reorder_strategy is ReorderStrategy.NONE:
            if self.verbose_flag:
                log.info('No reorder strategy is applied. Skipped...')
            return
        start = time.perf_counter()
        self.edge_index = reorder_with_strategy(
            reorder_strategy, self.edge_index, self.degrees_cpu, self.num_nodes)
        reorder_time = time.perf_counter() - start

        if self.verbose_flag:
            log.done(
                "# {} Reorder time (s): {:.3f}".format(reorder_strategy.name, reorder_time))

    def split_by_density(self, density, TILE_ROW, TILE_COL):
        self.dense_adj, self.sparse_adj = split_adj_list_by_density(
            self.adj_list, density, TILE_ROW, TILE_COL, self.num_nodes)
        self.A_dim = len(self.sparse_adj)
        if self.verbose_flag:
            log.done('=> Splitted the adjacency list!')
            # print('dense_adj:')
            # print(adj_list_to_adj_matrix(self.dense_adj, self.num_nodes))
            # print('sparse_adj:')
            # print(adj_list_to_adj_matrix(self.sparse_adj, self.num_nodes))

    def GNNA_gen_csr(self):
        ''' Generate CSR representation(torch.IntTensor) for the workload of GNNAdvisor.'''
        self.dense_row_pointers, self.dense_column_index = adj_list_to_csr(
            self.dense_adj)

    def gen_degrees_hat(self):
        '''
        Generate a new degrees array according to the updated edge_index.
        Note: degrees(D_hat) equals (D+I)^{-1/2}
        '''
        degrees = degrees_from_edge_index(self.edge_index, self.num_nodes)
        self.degrees_cpu = (1.0/torch.sqrt(torch.FloatTensor(
            list(map(func, degrees)))))

    def calc_d_a_d(self):
        '''
        Return D^A^D^ in array format of A_hat (for test).
        '''
        a_coo_matrix = adj_list_to_coo_matrix(self.adj_list, self.num_nodes)
        degrees_coo_matrix = diags([self.degrees_cpu.numpy()], [0])
        d_a_d = degrees_coo_matrix*a_coo_matrix*degrees_coo_matrix
        return d_a_d.toarray()

    def save_degrees_hat(self, path):
        '''
        Save D~^{-1/2} in '.npy' format for sparseRT.
        Return: file name of '.npy'.
        '''
        np.save(path, self.degrees_cpu.numpy())
        return path

    def save_AB(self, path):
        self.AB_coo_matrix = adj_list_to_coo_matrix(
            self.sparse_adj, self.num_nodes)
        log.info('SparseRT nnz count:{}'.format(
            self.AB_coo_matrix.count_nonzero()))
        save_npz(path, self.AB_coo_matrix)
        return path

    def save_for_sparsert(self, dest_dir):
        '''
        Save D^ and AB for sparseRT.
                degrees: D~^{-1/2} in '.npy' format
                AB: coo_matrix in '.npz' format
        Return: file names of degrees and AB
        '''
        if self.verbose_flag:
            log.info('# Saving D^ and AB for sparseRT')
        degrees_path = dest_dir+SPARSERT_DEGREES
        AB_path = dest_dir+SPARSERT_AB
        return self.save_degrees_hat(degrees_path), self.save_AB(AB_path)

    def edge_index_to_adj_list(self):
        self.adj_list = edge_index_to_adj_list(self.edge_index, self.num_nodes)

    def print_adj_matrix(self):
        print(edge_index_to_adj_matrix(self.edge_index, self.num_nodes))


if __name__ == '__main__':
    # path = osp.join("/home/yuke/.graphs/osdi-ae-graphs/", "cora.npz")
    # path = osp.join("/home/yuke/.graphs/osdi-ae-graphs/", "amazon0505.npz")
    path = osp.join(
        "/home/xiaosiyier/projects/OSDI21_AE/my-test-graphs/", "test_rabbit.txt")
    dataset = custom_dataset(path, 16, 10, load_from_txt=True, verbose=True)
    # reorder
    dataset.print_adj_matrix()
    dataset.reorder(ReorderStrategy.NONE)
    dataset.print_adj_matrix()
    dataset.reorder(ReorderStrategy.RANDOM)
    dataset.print_adj_matrix()
    dataset.degrees_cpu = degrees_from_edge_index(
        dataset.edge_index, dataset.num_nodes)
    dataset.reorder(ReorderStrategy.DEGREE)
    dataset.print_adj_matrix()
    dataset.reorder(ReorderStrategy.RABBIT)
    dataset.print_adj_matrix()
    dataset.reorder(ReorderStrategy.METIS)
    dataset.print_adj_matrix()
    dataset.plus_identity_matrix()
    print('plus I')
    dataset.print_adj_matrix()
    dataset.edge_index_to_adj_list()
    # split
    dataset.split_by_density(0, 2, 2)
    # GNNA
