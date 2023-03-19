import os
import os.path as osp
from utils import *
from cuda import cuda
import log


class SparseRTLayer():
    def __init__(self, BA_npy, inputInfo, C_dim, C_blocks, verbose):
        self.BA_npy = BA_npy

        self.A_dim = inputInfo.modeBarrier
        self.A_blocks = inputInfo.A_blocks
        self.B_dim = inputInfo.num_nodes
        self.Gy = inputInfo.Gy

        self.C_dim = C_dim
        self.C_blocks = C_blocks
        self.Block_size = (C_dim//C_blocks)*self.Gy
        self.verbose = verbose
        self.ctx = checkCudaErrors(cuda.cuCtxGetCurrent())

    def gen_ptx(self):
        gen_ptx_command = "python ../SparseRT/sparsednn/code_gen_ptx.py --A_dim {} --B_dim {} \
    --C_dim {} --A_blocks {} --C_blocks {} --Gy {} \
        --infile {} --outfile {}"\
            .format(self.A_dim, self.B_dim, self.C_dim, self.A_blocks, self.C_blocks, self.Gy, self.BA_npy, self.ptx_file)
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
        os.system(gen_cubin_command)
        if not osp.exists(self.cubin_file):
            log.fail('Failed to generate cubin!')
            exit(0)

    def prepare_dist_path(self, dist_path):
        '''Create dist path if not exists, and remove old products if any.
        '''
        if not osp.exists(dist_path):
            os.makedirs(dist_path)
        else:
            remove_files_if_exists(self.ptx_file, self.cubin_file)

    def gen_ptx_and_cubin(self):
        dist_without_suffix = self.BA_npy.replace(
            'npys', 'dist').split('.npy')[0]
        self.ptx_file = '{}_{}.ptx'.format(dist_without_suffix, self.C_dim)
        self.cubin_file = '{}_{}.cubin'.format(dist_without_suffix, self.C_dim)
        self.prepare_dist_path(osp.dirname(self.ptx_file))
        self.gen_ptx()
        self.gen_cubin()

    def get_func_handle(self):
        module = checkCudaErrors(
            cuda.cuModuleLoad(str2cstr(self.cubin_file)))
        self.cu_function = checkCudaErrors(cuda.cuModuleGetFunction(
            module, b'_Z2mmPPKfPPf'))
