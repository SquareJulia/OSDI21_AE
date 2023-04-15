#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# adjust
enable_rabbit = True
enable_sort_by_degree = True
rabbitRatio = 0.5
density = 0.001
A_tileDim = 80
B_tileDim = 80

A_blockDim = 8

partSize = 32
dimWorker = 16
warpPerBlock = 8        # only effective in manual model


manual_mode = True
verbose_mode = True
loadFromTxt = True


hidden = [16]  # TODO

dataDir = '../my-test-graphs'
# dataDir = '../osdi-ae-graphs'

dataset = [
    # ('g1.txt', 1000, 5),
    ('g100nodes.txt', 1000, 7)
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


for partsize in partsize_li:
    for hid in hidden:
        for data, d, c in dataset:
            command = "python GNNA_main.py --dataset {} --dim {} --hidden {} \
                        --classes {} --partSize {}  --warpPerBlock {}\
                        --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {} --dataDir {}"
            command = command.format(data, d, hid, c, partsize, warpPerBlock,
                                     manual_mode, verbose_mode, enable_rabbit, loadFromTxt, dataDir)
            # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')
            os.system(command)
