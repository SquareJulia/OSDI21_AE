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
    ('citeseer', 3703, 6, 0.0011),
    ('cora', 1433, 7, 0.0018),
    ('pubmed', 500, 3, 0.0003),
]
manual_mode = True

density_times = 1
A_tileDim = B_tileDim = 720

reorder_strategy = 'rabbit'


if is_pre:
    for data, d, c, base in dataset:
        density = base*density_times
        if data == 'pubmed':
            A_blockDim = 80
        else:
            A_blockDim = 16
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
        density = base*density_times
        if data == 'pubmed':
            A_blockDim = 80
        else:
            A_blockDim = 16

        command_train = "python GNNA_main.py \
        --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {}  --density {}\
                --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                    --verbose_mode {}".format(
            data, d,  c, loadFromTxt, dataDir, density,
            A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode)
        os.system(command_train)
        os.sync()
