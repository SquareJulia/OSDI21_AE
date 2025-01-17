import ctypes
from cuda import cuda
import os
import glob
from functools import reduce
from bisect import bisect_left
import torch
import log

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


def all_files(*files):
    for f in files:
        if not os.path.isfile(f):
            return False
    return True


def remove_files_if_exists(*files):
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


def factors(n):
    '''Return all the factors(ascending) of number n.'''
    ans = reduce(list.__add__,
                 ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))
    ans.sort()
    return ans


def first_ge(alist, threshold):
    '''Return first index i that alist[i]>=threshold
    '''
    return bisect_left(alist, threshold)


def compare_tensor(result, result_ref, info_head=''):
    ''' Compare the result with result_ref.
        Return True if they are almost equal.
    '''
    if result_ref is None or result is None:
        raise ValueError(
            "MUST compute result and result reference (CPU) first!!")

    # equs = torch.eq(result_ref, result.cpu())
    equs = torch.isclose(result_ref, result.cpu(), atol=1e-1)  # TODO
    correct = torch.sum(equs)
    # print('compute error ratio: {.3f}'.format(1 - correct/result_ref.numel()))
    equal = False
    if (1 - correct/result_ref.numel()) < 0.1:
        # log.done("# Verification PASSED")
        equal = True
    else:
        log.fail("# {} Verification FAILED".format(info_head))
        print('compute error ratio: {:.3f}'.format(
            1 - correct/result_ref.numel()))
    return equal


def save_adj_list(adj_list, path):
    ''' Save the adj_list(list[list[]]) in txt mode.
    '''
    txt_adj_list = ''
    for row in adj_list:
        txt_adj_list += ' '.join(str(f) for f in row)
        txt_adj_list += '\n'
    with open(path, 'w') as f:
        f.write(txt_adj_list)
