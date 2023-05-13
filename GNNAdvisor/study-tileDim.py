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

density_times = 10

# tileDim_li = [80, 240, 400, 560, 720, 880]
# tileDim_li = [880, 1200, 1520, 2400]
tileDim_li = [8, 16, 32, 64, 80, 160]


reorder_strategy = 'rabbit'

for tileDim in tileDim_li:
    A_tileDim = B_tileDim = tileDim
    if is_pre:
        for data, d, c, base in dataset:
            density = base*density_times
            if data == 'pubmed':
                A_blockDim = min(tileDim, 16)
            else:
                A_blockDim = min(tileDim, 32)

            command_pre = "python GNNA_main_pre.py \
            --dataset {} --dim {} --classes {}\
                --loadFromTxt {} --dataDir {} --density {}\
                    --A_tileDim {} --B_tileDim {} --A_blockDim {} --reorder_strategy {}\
                        --verbose_mode {} --manual_mode {}".format(
                data, d,  c, loadFromTxt, dataDir, density,
                A_tileDim, B_tileDim, A_blockDim, reorder_strategy, verbose_mode, manual_mode)
            os.system(command_pre)
    else:
        # print("******************************")
        # print("++ tileDim: {}".format(A_tileDim))
        # print("******************************")

        for data, d, c, base in dataset:
            if data == 'pubmed':
                A_blockDim = min(tileDim, 16)
            else:
                A_blockDim = min(tileDim, 32)
            density = base*density_times

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
