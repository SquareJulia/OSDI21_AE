import math
from utils import factors, first_ge
import log

MAX_BLOCK_SIZE = 1024

# package of input parameters


class inputProperty(object):
    def __init__(self,
                 hiddenDim=None,
                 dataset_obj=None,
                 manual_mode=True,
                 verbose=False,
                 enable_rabbit=None, enable_sort_by_degree=None, rabbitRatio=None,
                 density=None, A_tileDim=None, B_tileDim=None,
                 partSize=None, dimWorker=None, warpPerBlock=None, sharedMem=None,
                 A_blockDim=None, Gy_input=None, Gy_hidden=None, C_blocks_input=None, C_blocks_hidden=None):

        if dataset_obj is None:
            raise ValueError("Dataset object MUST SET !!!")

        if enable_rabbit and (rabbitRatio is None or rabbitRatio < 0 or rabbitRatio > 1):
            raise ValueError("rabbitRatio MUST BE BETWEEN 0 and 1 !!!")

        if density < 0 or density > 1:
            raise ValueError("density MUST BE BETWEEN 0 and 1 !!!")

        self.dataset_obj = dataset_obj

        self.num_nodes = dataset_obj.num_nodes
        self.num_classes = dataset_obj.num_classes
        self.avgNodeDegree = dataset_obj.avg_degree
        self.avgEdgeSpan = dataset_obj.avg_edgeSpan

        self.outputDim_input = hiddenDim
        self.outputDim_hidden = self.num_classes

        self.manual_mode = manual_mode
        self.verbose_flag = verbose
        self.state_set_input = False

        # Reorder
        self.enable_sort_by_degree=enable_sort_by_degree
        self.enable_rabbit=enable_rabbit
        self.reorder_by_degree_flag=False
        self.reorder_rabbit_flag=False
        self.rabbitBarrier = math.floor(self.num_nodes*rabbitRatio)  # TODO

        # Mode-divergence related
        self.density=density
        self.A_tileDim=A_tileDim
        self.B_tileDim=B_tileDim

        # GNNAdvisor related
        self.partSize = partSize
        self.dimWorker_input = dimWorker
        self.dimWorker_hidden = dimWorker
        self.warpPerBlock_input = warpPerBlock
        self.warpPerBlock_hidden = warpPerBlock

        self.partPtr = None
        self.part2Node = None

        self.MAX_warpPerBlock = 8
        self.MAX_sharedMemory = sharedMem * 0.4
        self.gap_smem = 100

        # SparseRT related
        self.Gy_input = Gy_input
        self.Gy_hidden = Gy_hidden
        self.C_blocks_input = C_blocks_input
        self.C_blocks_hidden = C_blocks_hidden

    def decider_dimWorker(self, outputDim):
        return min(outputDim, 16)

    def decider_sharedMemory(self, outputDim, layerName):
        sharedMemory = self.MAX_warpPerBlock * \
            (self.partSize * 4 + outputDim * 4 + self.gap_smem * 4)/1e3
        if self.verbose_flag:
            print(
                "{} shared memory (KB): {:.3f} ".format(layerName, sharedMemory))
        sharedMemory = min(sharedMemory, self.MAX_sharedMemory)
        if self.verbose_flag:
            print(
                "{} updated (KB): {:.3f}".format(layerName, sharedMemory))
        return sharedMemory

    def decider_warpPerBlock(self, outputDim, sharedMemory):
        warpPerBlock = int(
            sharedMemory * 1e3 / (self.partSize * 4 + outputDim * 4))
        return min(warpPerBlock, self.MAX_warpPerBlock)

    def decider_C_Blocks(self, outputDim):
        dim_factors = factors(outputDim)
        first_ge_49 = first_ge(dim_factors, 49)
        return 1 if first_ge_49 == len(dim_factors) else dim_factors[first_ge_49]

    # Not used any more
    def decider_A_Blocks(self, A_dim): 
        A_dim_factors = factors(A_dim)
        first_ge_8 = first_ge(A_dim_factors, 8)
        return 1 if first_ge_8 == len(A_dim_factors) else A_dim//A_dim_factors[first_ge_8]

    def decider_Gy(self, B_dim, C_dim, C_blocks):
        maxByBlockSize = MAX_BLOCK_SIZE//(C_dim//C_blocks)
        return min(max(1, B_dim//32), maxByBlockSize)

    def decider_SparseRT(self):
        '''Determine SparseRT related parameters.'''
        self.C_blocks_input = self.decider_C_Blocks(self.outputDim_input)
        self.C_blocks_hidden = self.decider_C_Blocks(self.outputDim_hidden)
        self.Gy_input = self.decider_Gy(
            self.num_nodes, self.outputDim_input, self.C_blocks_input)
        self.Gy_hidden = self.decider_Gy(
            self.num_nodes, self.outputDim_hidden, self.C_blocks_hidden)
    
    def decider_GNNA(self):
        '''Determine GNNAdvisor related parameters.'''
        self.partSize = int(self.avgNodeDegree)

        sharedMemory_input = self.decider_sharedMemory(
            self.outputDim_input, 'input-layer')
        sharedMemory_hidden = self.decider_sharedMemory(
            self.outputDim_hidden, 'hidden-layer')

        self.warpPerBlock_input = self.decider_warpPerBlock(
            self.outputDim_input, sharedMemory_input)
        self.warpPerBlock_hidden = self.decider_warpPerBlock(
            self.outputDim_hidden, sharedMemory_hidden)

        self.dimWorker_input = self.decider_dimWorker(
            self.outputDim_input)
        self.dimWorker_hidden = self.decider_dimWorker(
            self.outputDim_hidden)

    def decider(self):
        '''
        Determine the performance-related parameter here.
        manual_mode: using user-specified parameters
        auto_mode:   determining the parameters according to the GPU resources and scheduling performance consideration.
        '''
        # Determine whether to reorder by degree and/or by community-detection(rabbit).
        if self.manual_mode:
            self.dataset_obj.reorder_by_degree_flag = self.reorder_by_degree_flag=self.enable_sort_by_degree
            self.dataset_obj.reorder_rabbit_flag = self.reorder_rabbit_flag = self.enable_rabbit
        else:
            self.dataset_obj.reorder_by_degree_flag = self.reorder_by_degree_flag=True # TODO
            self.dataset_obj.reorder_rabbit_flag=self.reorder_rabbit_flag=math.sqrt(self.avgEdgeSpan) > math.sqrt(self.num_nodes)/100
        
        self.dataset_obj.degree_reorder()
        self.dataset_obj.rabbit_reorder()

        # Determine other parameters.
        if not self.manual_mode:
            self.decider_GNNA()
            self.decider_SparseRT()

            if self.verbose_flag:
                log.done("\n=> AUTO Decider Complete !!!\n")
        
        # Split the graph
        

    def set_input(self):
        '''
        Determine the performance-related parameter for input layer.
        Switch the parameter for input layer.
        '''
        self.dimWorker = self.dimWorker_input
        self.warpPerBlock = self.warpPerBlock_input
        self.state_set_input = True

        return self

    def set_hidden(self):
        '''
        Determine the performance-related parameter for hidden layer.
        Switch the parameter for hidden layer.
        '''
        self.dimWorker = self.dimWorker_hidden
        self.warpPerBlock = self.warpPerBlock_hidden
        self.state_set_input = False
        return self

    def print_param_SpRT(self):
        print('# modeBarrier: {}'.format(self.modeBarrier))
        print('# A_blocks: {}'.format(self.A_blocks))
        print('# Gy_input: {}'.format(self.Gy_input))
        print('# C_blocks_input: {}'.format(self.C_blocks_input))
        print('# Gy_hidden: {}'.format(self.Gy_hidden))
        print('# C_blocks_hidden: {}'.format(self.C_blocks_hidden))

    def print_param(self):
        if self.verbose_flag:
            if self.state_set_input:
                if self.manual_mode:
                    print("# manual INPUT partSize: {}".format(self.partSize))
                    print("# manual INPUT dimWorker: {}".format(self.dimWorker))
                    print("# manual INPUT warpPerBlock: {}".format(
                        self.warpPerBlock))
                else:
                    print("# auto INPUT partSize: {}".format(self.partSize))
                    print("# auto INPUT dimWorker: {}".format(self.dimWorker))
                    print("# auto INPUT warpPerBlock: {}".format(
                        self.warpPerBlock))
                    print("# auto INPUT reorder_flag: {}".format(
                        self.reorder_rabbit_flag))
                    print("# auto INPUT reorder_by_degree_flag: {}".format(
                        self.reorder_by_degree_flag))
            else:
                if self.manual_mode:
                    print("# manual HIDDEN partSize: {}".format(self.partSize))
                    print("# manual HIDDEN dimWorker: {}".format(self.dimWorker))
                    print("# manual HIDDEN warpPerBlock: {}".format(
                        self.warpPerBlock))
                else:
                    print("# auto HIDDEN partSize: {}".format(self.partSize))
                    print("# auto HIDDEN dimWorker: {}".format(self.dimWorker))
                    print("# auto HIDDEN warpPerBlock: {}".format(
                        self.warpPerBlock))
                    print("# auto HIDDEN reorder_flag: {}".format(
                        self.reorder_rabbit_flag))
                    print("# auto HIDDEN reorder_by_degree_flag: {}".format(
                        self.reorder_by_degree_flag))
