mv *.log logs/
mv *.csv logs/


# # ./study-reorderStrategy.py --preOrRun pre
# # ./study-reorderStrategy.py --preOrRun run|tee study_reorderStrategy.log
# # ./2_study2csv.py study_reorderStrategy.log

# # ./study-tileDim.py --preOrRun pre
# ./study-tileDim.py --preOrRun run|tee study_tileDim.log
# ./2_study2csv.py study_tileDim.log

./study-density.py --preOrRun pre
./study-density.py --preOrRun run|tee study_density.log
./2_study2csv.py study_density.log