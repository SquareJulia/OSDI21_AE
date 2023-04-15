#!/usr/bin/env python3
import torch
import math
import GNNAdvisor as GNNA
from param import *
from utils import compare_tensor
import log


# class ScatterAndGather(torch.autograd.Function):
#     '''
#     Basic Scatter and Gather kernel for GNN.
#     Graph is undirected.
#     '''
#     @staticmethod
#     def forward(ctx, X, inputInfo):
#         ctx.inputInfo = inputInfo
#         ctx.partSize, ctx.dimWorker, ctx.warpPerBlock = \
#             inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock
#         X_prime = GNNA.SAG(X, inputInfo.row_pointers, inputInfo.column_index,
#                            inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node,
#                            inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock)
#         return X_prime

#     @staticmethod
#     def backward(ctx, d_output):
#         inputInfo = ctx.inputInfo
#         d_input = GNNA.SAG(d_output, inputInfo.row_pointers, inputInfo.column_index,
#                            inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node,
#                            ctx.partSize, ctx.dimWorker, ctx.warpPerBlock)
#         return d_input


class GNNAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, inputInfo, SpRT_layer, a_hat_hat_for_test):
        ctx.save_for_backward(X, weight)
        ctx.inputInfo = inputInfo
        ctx.SpRT_layer = SpRT_layer
        ctx.partSize, ctx.dimWorker, ctx.warpPerBlock = \
            inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock

        ctx.a_hat_hat_for_test = a_hat_hat_for_test

        # print("[Foward]: {}\n{}\n{}\n{}\n{}".format(inputInfo.row_pointers, inputInfo.column_index,
        #                                 inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node))
        # print("[Foward]: partSize: {}, dimWorker: {}, warpPerBlock: {}".format(ctx.partSize, \
        #                                                     ctx.dimWorker, ctx.warpPerBlock))

        X_prime = GNNA.forward(X, weight, inputInfo.dataset_obj.dense_row_pointers, inputInfo.dataset_obj.dense_column_index,
                               inputInfo.dataset_obj.degrees_gpu, inputInfo.partPtr, inputInfo.part2Node,
                               inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock,
                               SpRT_layer.cu_function.getPtr(), SpRT_layer.A_blocks, SpRT_layer.C_blocks,
                               SpRT_layer.Block_size, SpRT_layer.ctx.getPtr())[0]

        X_prime_ref = torch.mm(a_hat_hat_for_test, torch.mm(X, weight).cpu())
        compare_tensor(X_prime, X_prime_ref)
        # if not compare_tensor(X_prime, X_prime_ref):
        #     print('X_prime:')
        #     torch.set_printoptions(precision=5, sci_mode=False)
        #     print(X_prime)
        #     print()
        #     print('X_prime_ref:')
        #     print(X_prime_ref)
        #     log.info('ref:')
        #     print('a_hat_hat:')
        #     print(a_hat_hat_for_test)
        #     print('X:')
        #     print(X)
        #     print('weight:')
        #     print(weight)
        # print(X.size())
        # print(weight.size())
        # X_prime = torch.mm(X, weight)
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weight = ctx.saved_tensors
        inputInfo = ctx.inputInfo
        SpRT_layer = ctx.SpRT_layer

        a_hat_hat_for_test = ctx.a_hat_hat_for_test
        # print("[Backward]: {}\n{}\n{}\n{}\n{}".format(inputInfo.row_pointers, inputInfo.column_index,         #                                 inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node))

        # print("[Backward]: partSize: {}, dimWorker: {}, warpPerBlock: {}".format(ctx.partSize, \
        #                                                     ctx.dimWorker, ctx.warpPerBlock))

        d_input, d_weight = GNNA.backward(d_output, X, weight, inputInfo.dataset_obj.dense_row_pointers, inputInfo.dataset_obj.dense_column_index,
                                          inputInfo.dataset_obj.degrees_gpu, inputInfo.partPtr, inputInfo.part2Node,
                                          ctx.partSize, ctx.dimWorker, ctx.warpPerBlock,
                                          SpRT_layer.cu_function.getPtr(), SpRT_layer.A_blocks, SpRT_layer.C_blocks,
                                          SpRT_layer.Block_size, SpRT_layer.ctx.getPtr())

        d_X_prime_ref = torch.mm(a_hat_hat_for_test, d_output.cpu())
        d_input_ref = torch.mm(d_X_prime_ref, weight.permute(1, 0).cpu())
        d_weight_ref = torch.mm(X.permute(1, 0).cpu(), d_X_prime_ref)
        if not compare_tensor(d_input, d_input_ref):
            log.fail('input error')
        if not compare_tensor(d_weight, d_weight_ref):
            log.fail('weight error')
        # if not compare_tensor(d_input, d_input_ref) or not compare_tensor(d_weight, d_weight_ref):
        #     torch.set_printoptions(precision=4, sci_mode=False)
        #     print('d_input:')
        #     print(d_input)
        #     print('d_input_ref:')
        #     print(d_input_ref)
        #     print('d_weight:')
        #     print(d_weight)
        #     print('d_weight_ref:')
        #     print(d_weight_ref)
        #     print('a_hat_hat:')
        #     print(a_hat_hat_for_test)
        #     print('d_output:')
        #     print(d_output)
        #     print('d_X_prime_ref:')
        #     print(d_X_prime_ref)

        # print(weight.size())
        # weight_p = weight.permute(1,0)
        # print(weight_p.size())
        # d_input =  torch.mm(d_X_prime, weight.permute(1,0));
        # d_weight = torch.mm(X.permute(1,0), d_X_prime);
        return d_input, d_weight, None, None, None


class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, SpRT_layer, a_hat_hat_for_test):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()
        self.SpRT_layer = SpRT_layer
        self.a_hat_hat_for_test = a_hat_hat_for_test

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, inputInfo):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        return GNNAFunction.apply(X, self.weights, inputInfo, self.SpRT_layer, self.a_hat_hat_for_test)
