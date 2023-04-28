#!/usr/bin/env python3
import re
import sys

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r").readlines()

dataset_li = []
time_li = []

forward_li = []
forward_sprt_li = []
forward_gnna_li = []
backward_li = []
backward_sprt_li = []
backward_gnna_li = []

forward_ms = 0.0
forward_cnt = 0
forward_sprt_ms = 0.0
forward_gnna_ms = 0.0
backward_ms = 0.0
backward_cnt = 0
backward_sprt_ms = 0.0
backward_gnna_ms = 0.0

line = 0
while line < len(fp):
    if "dataset=" in fp[line]:
        data = re.findall(
            r'dataset=.*?,', fp[line])[0].split('=')[1].replace(",", "").replace('\'', "")
        print(data)
        dataset_li.append(data)
        line += 1
        while(line < len(fp) and "Time (ms):" not in fp[line]):
            if "forward" in fp[line]:
                sprt, gnna, whole = list(
                    map(float, fp[line].split("forward")[1].rstrip("\n").split()))
                print('{} {} {}'.format(sprt, gnna, whole))
                forward_ms += whole
                forward_gnna_ms += gnna
                forward_sprt_ms += sprt
                forward_cnt += 1
            elif 'backward' in fp[line]:
                sprt, gnna, whole = list(
                    map(float, fp[line].split("backward")[1].rstrip("\n").split()))
                print('{} {} {}'.format(sprt, gnna, whole))
                backward_ms += whole
                backward_gnna_ms += gnna
                backward_sprt_ms += sprt
                backward_cnt += 1
            line += 1
        if "Time (ms):" in fp[line]:
            time = fp[line].split("Time (ms):")[1].rstrip("\n")
            print(time)
            time_li.append(time)
            forward_li.append('{:.3f}'.format(forward_ms/forward_cnt))
            forward_gnna_li.append(
                '{:.3f}'.format(forward_gnna_ms/forward_cnt))
            forward_sprt_li.append(
                '{:.3f}'.format(forward_sprt_ms/forward_cnt))
            backward_li.append('{:.3f}'.format(backward_ms/backward_cnt))
            backward_gnna_li.append('{:.3f}'.format(
                backward_gnna_ms/backward_cnt))
            backward_sprt_li.append('{:.3f}'.format(
                backward_sprt_ms/backward_cnt))
            forward_ms = forward_gnna_ms = forward_sprt_ms = 0
            backward_ms = backward_gnna_ms = backward_sprt_ms = 0
            forward_cnt = backward_cnt = 0
    line += 1

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
fout.write("dataset,Avg.Epoch (ms),Avg.forward (ms),Avg.backward(ms),Avg.forward-SparseRT(ms),Avg.forward-GNNA(ms),Avg.backward-SparseRT(ms),Avg.backward-GNNA(ms)\n")
for data, time, forward, backward, forward_sprt, forward_gnna, backward_sprt, backward_gnna in \
        zip(dataset_li, time_li, forward_li, backward_li, forward_sprt_li, forward_gnna_li, backward_sprt_li, backward_gnna_li):
    fout.write("{},{},{},{},{},{},{},{}\n".format(data, time, forward,
               backward, forward_sprt, forward_gnna, backward_sprt, backward_gnna))

fout.close()
fout.close()
