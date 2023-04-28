#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


loadFromTxt = False
# verbose_mode = False
verbose_mode = True
# dataDir = '../my-test-graphs'
dataDir = '../osdi-ae-graphs'

manual_mode = False


dataset = [
    # ('g5nodes.txt', 1000, 2)
    # ('g100nodes.txt', 1000, 6)
    # ('g100nodes_2.txt', 1000, 6)
    ('citeseer', 3703, 6),
    # ('g50nodes.txt', 1000, 6)
    # ('g10nodes.txt', 1000, 6)
]
density = 0.1
density = 0.00083
A_tileDim = 80
B_tileDim = 80
A_blockDim = 8
reorder_strategy = 'rabbit'


for data, d, c in dataset:
    command_pre = "python GNNA_main_pre.py \
    --dataset {} --dim {} --classes {}\
        --loadFromTxt {} --dataDir {} --density {}\
            --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                --verbose_mode {} --manual_mode {}".format(
        data, d,  c, loadFromTxt, dataDir, density,
        A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode, manual_mode)
    # os.system(command_pre)

    command_train = "python GNNA_main.py \
    --dataset {} --dim {} --classes {}\
        --loadFromTxt {} --dataDir {}  --density {}\
            --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                --verbose_mode {}".format(
        data, d,  c, loadFromTxt, dataDir, density,
        A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode)
    os.system(command_train)
