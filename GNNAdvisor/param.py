import math
from utils import factors, first_ge

# package of input parameters


class inputProperty(object):
    def __init__(self, row_pointers=None, column_index=None, degrees=None,
                 partSize=None, dimWorker=None, warpPerBlock=None,
                 sharedMem=None,
                 hiddenDim=None,
                 dataset_obj=None,
                 enable_rabbit=False,
                 manual_mode=True,
                 verbose=False,
                 sparseRTRatio=1.0,
                 A_blocks=None,
                 Gy=None,
                 C_blocks_input=None,
                 C_blocks_hidden=None):

        if dataset_obj is None:
            raise ValueError("Dataset object MUST SET !!!")

        if sparseRTRatio < 0 or sparseRTRatio > 1:
            raise ValueError("sparseRTRation MUST BE BETWEEN 0 and 1 !!!")

        self.dataset_obj = dataset_obj

        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees

        self.num_nodes = dataset_obj.num_nodes
        self.num_classes = dataset_obj.num_classes
        self.avgNodeDegree = dataset_obj.avg_degree
        self.avgEdgeSpan = dataset_obj.avg_edgeSpan

        self.partSize = partSize
        self.dimWorker = dimWorker
        self.warpPerBlock = warpPerBlock

        self.dimWorker_input = dimWorker
        self.dimWorker_hidden = dimWorker
        self.warpPerBlock_input = warpPerBlock
        self.warpPerBlock_hidden = warpPerBlock
        self.outputDim_input = hiddenDim
        self.outputDim_hidden = self.num_classes

        self.manual_mode = manual_mode
        self.enable_rabbit = enable_rabbit
        self.verbose_flag = verbose
        self.state_set_input = False
        self.reorder_status = False

        self.MAX_warpPerBlock = 8
        self.MAX_sharedMemory = sharedMem * 0.4
        self.gap_smem = 100

        self.partPtr = None
        self.part2Node = None
        # SparseRT related
        self.modeBarrier = math.floor(self.num_nodes*sparseRTRatio)
        self.A_blocks = A_blocks
        self.Gy = Gy
        self.C_blocks_input = C_blocks_input
        self.C_blocks_hidden = C_blocks_hidden

    def deciderOfDimWorker(self, outputDim):
        return min(outputDim, 16)

    def deciderOfSharedMemory(self, outputDim, layerName):
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

    def deciderOfWarpPerBlock(self, outputDim, sharedMemory):
        warpPerBlock = int(
            sharedMemory * 1e3 / (self.partSize * 4 + outputDim * 4))
        return min(warpPerBlock, self.MAX_warpPerBlock)

    def deciderOfCBlocks(self, outputDim):
        dim_factors = factors(outputDim)
        first_ge_49 = first_ge(dim_factors, 49)
        return 1 if first_ge_49 == len(dim_factors) else dim_factors[first_ge_49]

    def deciderOfABlocks(self, A_dim):
        A_dim_factors = factors(A_dim)
        first_ge_8 = first_ge(A_dim_factors, 8)
        return 1 if first_ge_8 == len(A_dim_factors) else A_dim//A_dim_factors[first_ge_8]

    def deciderOfGy(self, B_dim):
        return max(1, B_dim//32)

    def deciderOfSparseRT(self):
        '''Determine SparseRT related parameters.
        # TODO
        '''
        self.A_blocks = self.deciderOfABlocks(self.modeBarrier)
        self.C_blocks_input = self.deciderOfCBlocks(self.outputDim_input)
        self.C_blocks_hidden = self.deciderOfCBlocks(self.outputDim_hidden)
        self.Gy = self.deciderOfGy(self.num_nodes)

    def decider(self):
        '''
        Determine the performance-related parameter here.
        manual_mode: using user-specified parameters
        auto_mode:   determining the parameters according to the GPU resources and scheduling performance consideration.
        '''

        if self.manual_mode:
            if self.enable_rabbit:
                self.dataset_obj.reorder_flag = True
                self.dataset_obj.rabbit_reorder()
                self.reorder_status = True
                self.row_pointers = self.dataset_obj.row_pointers
                self.column_index = self.dataset_obj.column_index
            else:
                self.dataset_obj.reorder_flag = False
                self.reorder_status = False

            if self.verbose_flag:
                print("\n=> MANUAL Config Complete !!!\n")
        else:
            # Determine the neighbor partitioning.
            self.partSize = int(self.avgNodeDegree)

            sharedMemory_input = self.deciderOfSharedMemory(
                self.outputDim_input, 'input-layer')
            sharedMemory_hidden = self.deciderOfSharedMemory(
                self.outputDim_hidden, 'hidden-layer')

            # Determine the warpPerBlock for input and hidden layer.
            self.warpPerBlock_input = self.deciderOfWarpPerBlock(
                self.outputDim_input, sharedMemory_input)
            self.warpPerBlock_hidden = self.deciderOfWarpPerBlock(
                self.outputDim_hidden, sharedMemory_hidden)

            # Determine the dimWorker_input for input layer and hidden layer.
            self.dimWorker_input = self.deciderOfDimWorker(
                self.outputDim_input)
            self.dimWorker_hidden = self.deciderOfDimWorker(
                self.outputDim_hidden)

            if self.enable_rabbit:
                # Determine whether to reorder a graph.
                if math.sqrt(self.avgEdgeSpan) > math.sqrt(self.num_nodes)/100:
                    self.dataset_obj.reorder_flag = True
                    self.reorder_status = True
                else:
                    self.dataset_obj.reorder_flag = False
                    self.reorder_status = False

                self.dataset_obj.rabbit_reorder()

            self.deciderOfSparseRT()

            if self.verbose_flag:
                print("\n=> AUTO Decider Complete !!!\n")

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
                        self.reorder_status))
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
                        self.reorder_status))
