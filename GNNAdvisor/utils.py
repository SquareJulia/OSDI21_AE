import ctypes
from cuda import cuda
import os
import glob

CUDA_SUCCESS = 0


def str2cstr(str):
    '''
    Conver Python string to (char *) in C.
    '''
    b_str = str.encode('utf-8')
    return b_str


# from example.common.helper_cuda
def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(
            result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def remove_files_if_exists(*files):
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
