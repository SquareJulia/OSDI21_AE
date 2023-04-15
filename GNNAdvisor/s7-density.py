#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# density_li = [0, 0.0005, 0.001, 0.002, 0.01, 0.1, 1]
density_li = [1]

dataset = [
    ('g100nodes.txt', 1000, 7),
    # ('citeseer', 3703, 6),
    # ('amazon0505', 96, 22),
    # ('artist', 100, 12),
    # ('com-amazon', 96, 22),
    # ('soc-BlogCatalog', 128, 39),
    # ('amazon0601', 96, 22),
]

loadFromTxt = True
dataDir = '../my-test-graphs'

for density in density_li:
    print("******************************")
    print("++ density: {}".format(density))
    print("******************************")
    for data, d, c in dataset:
        print("{}---density: {}".format(data, density))
        print("=================")
        command = "python GNNA_main.py --dataset {} --dim {} --classes {}\
            --loadFromTxt {} --dataDir {} --density {}".format(
            data, d,  c, loadFromTxt, dataDir, density)
        os.system(command)
