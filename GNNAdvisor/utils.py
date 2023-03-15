import ctypes
from cuda import cuda

CUDA_SUCCESS = 0


def str2cstr(str):
    '''
    Conver Python string to (char *) in C.
    '''
    b_str = str.encode('utf-8')
    return b_str


def check_cu_result(func_name, cu_result):
    if cu_result != CUDA_SUCCESS:
        _, cu_string = cuda.cuGetErrorString(cu_result)
        print("%s failed with error code %d: %s" %
              (func_name, cu_result, cu_string))
