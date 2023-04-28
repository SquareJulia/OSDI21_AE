#!/usr/bin/env python3
import re
import sys

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r").readlines()

dataset_li = []
time_li = []


line = 0
while line < len(fp):
    if "dataset=" in fp[line]:
        data = re.findall(
            r'dataset=.*?,', fp[line])[0].split('=')[1].replace(",", "").replace('\'', "")
        print(data)
        dataset_li.append(data)
    if "Time (ms):" in fp[line]:
        time = fp[line].split("Time (ms):")[1].rstrip("\n")
        print(time)
        time_li.append(time)
    line += 1

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
fout.write("dataset,Avg.Epoch (ms)\n")
for data, time in \
        zip(dataset_li, time_li):
    fout.write("{},{}\n".format(data, time))

fout.close()
