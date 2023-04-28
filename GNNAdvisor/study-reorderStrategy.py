#!/usr/bin/env python3
import os
import argparse
os.environ["PYTHONWARNINGS"] = "ignore"
parser = argparse.ArgumentParser()
parser.add_argument("--preOrRun", type=str, choices=['pre', 'run'],
                    default="pre", help="preprocess or run: pre/run")

args = parser.parse_args()
is_pre = args.preOrRun == 'pre'

loadFromTxt = False
verbose_mode = False
dataDir = '../osdi-ae-graphs'
dataset = [
    ('citeseer', 3703, 6),
    ('cora', 1433, 7),
    ('pubmed', 500, 3),
]
manual_mode = True

density = 0.005
A_tileDim = B_tileDim = 80
A_blockDim = 16

# reorder_strategy_li = ['METIS']
reorder_strategy_li = ['None', 'random', 'degree', 'rabbit']

for reorder_strategy in reorder_strategy_li:
    if is_pre:
        for data, d, c in dataset:
            command_pre = "python GNNA_main_pre.py \
            --dataset {} --dim {} --classes {}\
                --loadFromTxt {} --dataDir {} --density {}\
                    --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                        --verbose_mode {} --manual_mode {}".format(
                data, d,  c, loadFromTxt, dataDir, density,
                A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode, manual_mode)
            os.system(command_pre)
    else:
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
