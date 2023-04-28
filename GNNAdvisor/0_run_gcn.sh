mkdir logs
mv *.csv logs/
mv *.log logs/
./bench_GCN.py --preOrRun pre
for i in `seq 1 20`
    do
        ./bench_GCN.py --preOrRun run|tee -a GCN.log
    done
./1_log2csv.py GCN.log