#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# density = 1.0
# A_tileDim = 4
# B_tileDim = 4
# A_blockDim = 4

# A_tileDim = 32
# B_tileDim = 32
# A_blockDim = 16

A_tileDim = 4
B_tileDim = 4
A_blockDim = 2

# density_li = [0, 0.001, 0.01, 1]
density_li = [1]

dataset = [
    # ('g5nodes.txt', 1000, 2)
    # ('g100nodes.txt', 1000, 6)
    # ('g100nodes_2.txt', 1000, 6)
    ('citeseer', 3703, 6),
    # ('g50nodes.txt', 1000, 6)
    # ('g10nodes.txt', 1000, 6)
]

loadFromTxt = False
loadFromTxt = True
# verbose_mode = False
verbose_mode = True
# dataDir = '../my-test-graphs'
dataDir = '../osdi-ae-graphs'
reorder_strategy = 'rabbit'

for density in density_li:
    for data, d, c in dataset:
        command_pre = "python GNNA_main_pre.py \
        --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {} --density {}\
                --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                    --verbose_mode {} --manual_mode False".format(
            data, d,  c, loadFromTxt, dataDir, density,
            A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode)
        os.system(command_pre)

        command_train = "python GNNA_main.py \
        --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {}  \
                --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                    --verbose_mode {}".format(
            data, d,  c, loadFromTxt, dataDir,
            A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode)
        os.system(command_train)
