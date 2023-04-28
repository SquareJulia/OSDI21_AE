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
verbose_mode = True
dataDir = '../osdi-ae-graphs'
dataset = [
    ('citeseer'	        , 3703	    , 6   ),  
        ('cora' 	        , 1433	    , 7   ),  
        ('pubmed'	        , 500	    , 3   ),
]
manual_mode = True

density_li = [0.001, 0.005, 0.01, 0.012, 0.02, 0.1]


A_tileDim = 80
B_tileDim = 80
A_blockDim = 16
reorder_strategy = 'rabbit'

for density in density_li:
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
        print("++ density: {}".format(density))
        print("******************************")

        for data, d, c in dataset:
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
