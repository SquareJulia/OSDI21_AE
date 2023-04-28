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
verbose_mode = True
dataDir = '../osdi-ae-graphs'
dataset = [
    ('citeseer', 3703, 6),
    ('cora', 1433, 7),
    ('pubmed', 500, 3),
]
manual_mode = True

density = 0.001

tileDim_li = [160, 240, 320, 400, 480, 560, 640]


A_blockDim = 16
reorder_strategy = 'rabbit'

for tileDim in tileDim_li:
    A_tileDim = B_tileDim = tileDim
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
        print("++ tileDim: {}".format(A_tileDim))
        print("******************************")

        for data, d, c in dataset:
            print("{}---tileDim: {}".format(data, A_tileDim))
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
