#!/usr/bin/env python3
import torch
import math
import GNNAdvisor as GNNA
from param import *

class GNNAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, inputInfo):
        ctx.save_for_backward(X, weight)
        ctx.inputInfo = inputInfo
        # print("partSize: {}, dimWorker: {}, warpPerBlock: {}".format(inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock))
        X_prime = GNNA.forward(X, weight, inputInfo.row_pointers, inputInfo.column_index, 
                                inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node, \
                                inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock)[0]
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weight = ctx.saved_tensors
        inputInfo = ctx.inputInfo

        d_input, d_weight = GNNA.backward(d_output, X, weight, inputInfo.row_pointers, inputInfo.column_index, 
                                        inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node,
                                        inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock)

        return d_input, d_weight, None

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()

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
        return GNNAFunction.apply(X, self.weights, inputInfo)


class GNNAFunction_GIN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, inputInfo, eplison):

        X_prime, X_agg = GNNA.forward_gin(X, weight, inputInfo.row_pointers, inputInfo.column_index, 
                                        eplison, inputInfo.partPtr, inputInfo.part2Node, 
                                        inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock)

        ctx.save_for_backward(X_agg, weight)
        ctx.inputInfo = inputInfo
        ctx.eplison = eplison

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        
        X, weights,  = ctx.saved_tensors
        inputInfo = ctx.inputInfo

        d_input, d_weights = GNNA.backward_gin(d_output, X, weights, inputInfo.row_pointers, inputInfo.column_index,
                                               ctx.eplison, inputInfo.partPtr, inputInfo.part2Node,
                                                inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock)
        
        return d_input, d_weights, None, None

class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.eplison = 0.5
        self.reset_parameters()

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
        return GNNAFunction_GIN.apply(X, self.weights, inputInfo, self.eplison)