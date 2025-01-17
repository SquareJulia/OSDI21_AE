#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


run_GCN = True              # whether to run GCN model.
# whether to enable rabbit reordering in auto and manual mode.
enable_rabbit = True
# whether to use the manually configure the setting.
manual_mode = False
# whether to printout more information such as the layerwise parameter.
verbose_mode = True
# loadFromTxt = False       # whether to load data from a plain txt file.
loadFromTxt = True

if run_GCN:
    model = 'gcn'
    warpPerBlock = 8        # only effective in manual model
    hidden = [16]  # TODO
else:
    model = 'gin'
    # only effective in manual model 2 for citeseer 6 for remaining datasets
    warpPerBlock = 2
    hidden = [64]

partsize_li = [32]          # only effective in manual model

dataset = [
    # ('g1.txt', 1000, 5),
    ('g5nodes.txt', 1000, 2)
    # ('citeseer', 3703, 6),
    # ('cora', 1433, 7),
    # ('pubmed'	        , 500	    , 3   ),
    # ('ppi'	            , 50	    , 121 ),

    # ('PROTEINS_full'             , 29       , 2) ,
    # ('OVCAR-8H'                  , 66       , 2) ,
    # ('Yeast'                     , 74       , 2) ,
    # ('DD'                        , 89       , 2) ,
    # ('TWITTER-Real-Graph-Partial', 1323     , 2) ,
    # ('SW-620H'                   , 66       , 2) ,

    # ( 'amazon0505'               , 96	, 22),
    # ( 'artist'                   , 100  , 12),
    # ( 'com-amazon'               , 96	, 22),
    # ( 'soc-BlogCatalog'	       	 , 128  , 39),
    # ( 'amazon0601'  	         , 96	, 22),
]
dataDir = '../my-test-graphs'
# dataDir = '../osdi-ae-graphs'

for partsize in partsize_li:
    for hid in hidden:
        for data, d, c in dataset:
            command = "python GNNA_main.py --dataset {} --dim {} --hidden {} \
                        --classes {} --partSize {} --model {} --warpPerBlock {}\
                        --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {} --dataDir {}"
            command = command.format(data, d, hid, c, partsize, model, warpPerBlock,
                                     manual_mode, verbose_mode, enable_rabbit, loadFromTxt, dataDir)
            # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')
            os.system(command)
