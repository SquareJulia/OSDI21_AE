#!/usr/bin/env python3
import argparse
import os
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
    # ('citeseer', 3703, 6, 0.0011),
    # ('cora', 1433, 7, 0.0018),
    ('pubmed', 500, 3, 0.0003),
]
manual_mode = True

density = 1

reorder_strategy = 'rabbit'

A_tileDim = B_tileDim = 32
A_blockDim = 32

if is_pre:
    for data, d, c, base in dataset:

        command_pre = "python GNNA_main_pre.py \
        --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {} --density {}\
                --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                    --verbose_mode {} --manual_mode {}".format(
            data, d,  c, loadFromTxt, dataDir, density,
            A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode, manual_mode)
        os.system(command_pre)
else:

    for data, d, c, base in dataset:
        print("{}---density: {}".format(data, density))
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
