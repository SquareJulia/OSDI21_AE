mv *.log new-logs/
mv *.csv new-logs/

# ./s7-4_1_neighbor_partitioning.py| tee study_partition.log
# ./2_study2csv.py study_partition.log 

# ./s7-4_2_dimension_partitiong.py| tee study_dimWorker.log
# ./2_study2csv.py study_dimWorker.log 

# ./s7-4_3_node_renumbering.py| tee study_nodeReordering.log
# ./2_study2csv.py study_nodeReordering.log

# ./s7-5_1_hidden_dimension.py| tee study_hiddenDimension.log
# ./2_study2csv.py study_hiddenDimension.log 

./s7-density.py|tee study_density.log