#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


loadFromTxt = False
# verbose_mode = False
verbose_mode = True
# dataDir = '../my-test-graphs'
dataDir = '../osdi-ae-graphs'

manual_mode = True

density = 0.01

tileDim_li = [160, 240, 320, 400, 480, 560, 640]


dataset = [
    # ('g5nodes.txt', 1000, 2)
    # ('g100nodes.txt', 1000, 6)
    # ('g100nodes_2.txt', 1000, 6)
    ('citeseer', 3703, 6),
    # ('g50nodes.txt', 1000, 6)
    # ('g10nodes.txt', 1000, 6)
]
A_blockDim = 16
reorder_strategy = 'rabbit'

for tileDim in tileDim_li:
    A_tileDim = B_tileDim = tileDim
    for data, d, c in dataset:
        command_pre = "python GNNA_main_pre.py \
        --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {} --density {}\
                --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                    --verbose_mode {} --manual_mode {}".format(
            data, d,  c, loadFromTxt, dataDir, density,
            A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode, manual_mode)
        # os.system(command_pre)
        os.sync()

        print('tileDim:{}'.format(tileDim))

        command_train = "python GNNA_main.py \
        --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {}  --density {}\
                --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                    --verbose_mode {}".format(
            data, d,  c, loadFromTxt, dataDir, density,
            A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode)
        os.system(command_train)
