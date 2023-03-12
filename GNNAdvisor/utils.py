import ctypes
def str2cstr(str):
    '''
    Conver Python string to (char *) in C.
    '''
    b_str=str.encode('utf-8')
    return b_str