import os
import os.path as osp
from utils import *
from cuda import cuda
import log
import time


class SparseRTLayer():
    def __init__(self, degrees_file, AB_file, inputInfo, C_dim, C_blocks, Gy, verbose):
        self.degrees_file = degrees_file
        self.AB_file = AB_file

        self.A_dim = inputInfo.dataset_obj.A_dim
        A_blockDim = inputInfo.A_blockDim
        assert self.A_dim % A_blockDim == 0
        self.A_blocks = self.A_dim//A_blockDim
        self.B_dim = inputInfo.num_nodes

        self.C_dim = C_dim
        self.C_blocks = C_blocks
        self.Gy = Gy
        self.Block_size = (C_dim//C_blocks)*self.Gy
        self.verbose = verbose
        self.ctx = None

        self.ptx_file = ''
        self.cubin_file = ''
        self.cu_function = None

    def gen_ptx(self):
        gen_ptx_command = "python ../SparseRT/sparsednn/code_gen_ptx.py --A_dim {} --B_dim {} \
    --C_dim {} --A_blocks {} --C_blocks {} --Gy {} \
        --degrees_file {} --AB_file {} --outfile {}"\
            .format(self.A_dim, self.B_dim, self.C_dim, self.A_blocks, self.C_blocks, self.Gy, self.degrees_file, self.AB_file, self.ptx_file)
        if self.verbose:
            log.info('+ generating ptx with C_dim:{}...'.format(self.C_dim))
            print(gen_ptx_command)
        os.system(gen_ptx_command)
        if not osp.exists(self.ptx_file):
            log.fail('Failed to generate ptx!')
            exit(0)

    def gen_cubin(self):
        gen_cubin_command = "ptxas -arch=sm_70 {} -o {}".format(
            self.ptx_file, self.cubin_file)
        if self.verbose:
            log.info('+ generating cubin with C_dim:{}...'.format(self.C_dim))
            print(gen_cubin_command)
        start = time.perf_counter()
        os.system(gen_cubin_command)
        if not osp.exists(self.cubin_file):
            log.fail('Failed to generate cubin!')
            exit(0)
        if self.verbose:
            log.info('# Generate .cubin(s): {:.3f}'.format(
                time.perf_counter()-start))

    def prepare_dist_path(self, dist_path):
        '''Create dist path if not exists, and remove old products if any.
        '''
        if not osp.exists(dist_path):
            os.makedirs(dist_path)
        else:
            remove_files_if_exists(self.ptx_file, self.cubin_file)

    def gen_ptx_and_cubin(self, inputInfo):
        dist_without_suffix = self.AB_file.replace(
            'data', 'dist').split('.npz')[0]
        d_temp = 'd' if inputInfo.reorder_by_degree_flag else 'x'
        r_temp = 'r{}'.format(
            inputInfo.rabbitBarrier) if inputInfo.reorder_rabbit_flag else 'x'
        template = '{}_{}_{}{}_{}_{}'.format(
            dist_without_suffix, self.C_dim, d_temp, r_temp, inputInfo.density, inputInfo.A_tileDim)  # TODO:use template
        self.ptx_file = '{}.ptx'.format(template)
        self.cubin_file = '{}.cubin'.format(template)
        self.prepare_dist_path(osp.dirname(self.ptx_file))
        self.gen_ptx()
        self.gen_cubin()

    def get_func_handle(self):
        module = checkCudaErrors(
            cuda.cuModuleLoad(str2cstr(self.cubin_file)))
        self.cu_function = checkCudaErrors(cuda.cuModuleGetFunction(
            module, b'_Z2mmPPKfPPf'))

    def get_ctx(self):
        self.ctx = checkCudaErrors(cuda.cuCtxGetCurrent())
