mkdir logs
mv *.csv logs/
mv *.log logs/
./bench_GCN.py --preOrRun pre
./bench_GCN.py --preOrRun run|tee GCN.log
./1_log2csv.py GCN.log