import torch


def is_basic_type(x):
    types = [str, int, float, bool, list]
    for c in types:
        if isinstance(x, c):
            return True
    return False


def print_obj(obj):
    for k, v in obj.__dict__.items():
        if is_basic_type(k):
            print(' # {}: {}'.format(k, v))
        else:
            print(' # {}:'.format(k))
            print('   {}:'.format(v))


dataset = torch.load('dataset.pt')
inputInfo = torch.load('inputInfo.pt')

print(dataset.print_adj_matrix())
# print(dataset == inputInfo.dataset_obj)
# print('---------------------')
# print_obj(dataset)
# print('---------------------')
# print_obj(inputInfo)
