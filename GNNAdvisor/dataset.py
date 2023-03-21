#!/usr/bin/env python3
import torch
import numpy as np
import time
import dgl
import os.path as osp
import os

from scipy.sparse import *
import rabbit
from constants import SPARSERT_MATERIALS_NPYS_DIR, SPARSERT_MATERIALS_DIST_DIR
import log


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

        self.reorder_flag = False
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

            self.num_nodes = max(self.nodes) + 1
            self.edge_index = np.stack([src_li, dst_li])

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

        # plus Identity matrix
        src_li = np.concatenate((src_li, [i for i in range(self.num_nodes)]))
        dst_li = np.concatenate((dst_li, [i for i in range(self.num_nodes)]))
        self.edge_index = np.stack([src_li, dst_li])
        self.num_edges += self.num_nodes

        self.val = [1] * self.num_edges
        self.generate_csr_and_degrees()

    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, dim).cuda()

    def init_labels(self, num_class):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.ones(self.num_nodes).long().cuda()

    def rabbit_reorder(self):
        '''
        If the decider set this reorder flag,
        then reorder and rebuild a graph CSR.
        otherwise skipped this reorder routine.
        Called from external.
        '''
        if not self.reorder_flag:
            if self.verbose_flag:
                log.info("Reorder flag is not set. Skipped...")
        else:
            if self.verbose_flag:
                log.info("Reorder flag is set. Continue...")
                print("Original edge_index:\n", self.edge_index)
            start = time.perf_counter()
            self.edge_index = rabbit.reorder(torch.IntTensor(self.edge_index))
            reorder_time = time.perf_counter() - start

            if self.verbose_flag:
                log.done("# Reorder time (s): {}".format(reorder_time))
                print("Reordered edge_index:\n", self.edge_index)

            self.generate_csr_and_degrees()

    def generate_csr_and_degrees(self):
        '''
        Generate a new graph CSR and degrees array according to the updated edge_index.
        '''
        start = time.perf_counter()

        scipy_coo = coo_matrix((self.val, self.edge_index),
                               shape=(self.num_nodes, self.num_nodes))
        self.a_hat = scipy_coo.toarray().astype('float32')
        scipy_csr = scipy_coo.tocsr()
        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)
        build_csr = time.perf_counter() - start

        # Re-generate degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = (1.0/torch.sqrt(torch.FloatTensor(
            list(map(func, degrees))))).cuda()

        # print('-------')
        # print('degrees:')
        # print(self.degrees)
        # print('-------')
        # print('A_hat:')
        # print(self.a_hat)
        # print('-------')
        # print('column_index:')
        # print(self.column_index)
        # print('-------')
        # print('row_pointers:')
        # print(self.row_pointers)

        if self.verbose_flag:
            print("# Re-Build CSR (s): {:.3f}".format(build_csr))

    def a_times_d(self, modeBarrier=None):
        '''
        Return D^A^D^ for [0, modeBarrier) lines of A_hat (for SparseRT).
        For GNNA, this step will be conducted in the GPU kernel.
        '''
        modeBarrier = modeBarrier if modeBarrier is not None else self.num_nodes
        a_hat_hat = self.a_hat[0:modeBarrier].copy()
        for row in range(modeBarrier):
            for col_idx_idx in range(self.row_pointers[row], self.row_pointers[row+1]):
                col = self.column_index[col_idx_idx].item()
                a_hat_hat[row][col] *= self.degrees[row].item()
                a_hat_hat[row][col] *= self.degrees[col].item()
        return a_hat_hat

    def save_transposed_sparse_matrix_npy(self, modeBarrier):
        '''
        Save [0,modeBarrier) lines of D^A^D^ in '.npy' format for sparseRT.
        Return: file name of '.npy'.
        '''
        if modeBarrier == 0:
            return ''
        if self.verbose_flag:
            log.info('# Saving transposed A^(i.e.D^A^D^) for sparseRT')
            # print('=== original A^:')
            # torch.set_printoptions(precision=2)
            # print(self.a_hat)
        a_hat_hat = self.a_times_d(modeBarrier)
        # if self.verbose_flag:
        # print('=== D^A^D^:')
        # torch.set_printoptions(precision=2)
        # print(a_hat_hat)

        npy_dir = '{}{}'.format(
            SPARSERT_MATERIALS_NPYS_DIR, osp.dirname(self.path).split('../')[1])
        if not osp.exists(npy_dir):
            os.makedirs(npy_dir)
        npy_basename = '{}_{}.npy'.format(
            osp.basename(self.path).split('.')[0], modeBarrier)
        npy_path = osp.join(npy_dir, npy_basename)
        if os.path.isfile(npy_path):
            os.remove(npy_path)

        np.save(npy_path, np.transpose(a_hat_hat))
        if self.verbose_flag:
            log.done('transposed A^ saved at:{}'.format(npy_path))
        return npy_path


if __name__ == '__main__':
    # path = osp.join("/home/yuke/.graphs/osdi-ae-graphs/", "cora.npz")
    path = osp.join("/home/yuke/.graphs/osdi-ae-graphs/", "amazon0505.npz")
    dataset = custom_dataset(path, 16, 10, load_from_txt=False)
    dataset.reorder_flag = True
    dataset.rabbit_reorder()
