#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


loadFromTxt = False
verbose_mode = False
dataDir = '../osdi-ae-graphs'

manual_mode = True

density = 0.01
A_tileDim = B_tileDim = 560
A_blockDim = 16


dataset = [
    ('citeseer', 3703, 6)
]
reorder_strategy_li = ['None', 'random', 'degree', 'rabbit']
# reorder_strategy_li = ['None', 'random', 'degree', 'rabbit', 'METIS']

for reorder_strategy in reorder_strategy_li:
    print("******************************")
    print("++ reorderStrategy: {}".format(reorder_strategy))
    print("******************************")

    for data, d, c in dataset:
        print("{}---reorderStrategy: {}".format(data, reorder_strategy))
        print("=================")
        command_train = "python GNNA_main.py \
        --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {}  --density {}\
                --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                    --verbose_mode {}".format(
            data, d,  c, loadFromTxt, dataDir, density,
            A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode)
        os.system(command_train)
        os.sync()
