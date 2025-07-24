#source activate dg_rdg
echo "change to conda in gpu server"
echo "----begin identify moddel train------"


#for ((i=1; i<=5; i++));do for f in 900 700 500 300 100 50; do
exp=result
config_file=best_config2.yml

## for test 
test_data=./data/3.7k_snp.pickle.test
logger_file=$exp/test.log
snp_mark_file=$exp/snp_tokens
test_result_file=$exp/test_result
checkpoint=$exp/model_checkpoint


echo "--begin to trian model"

python ./rdc_identify_model.py  --method train_with_stop_snp  --conf $config_file 

echo "----begin identify moddel test------"

python ./rdc_classify_test.py --checkpoint $checkpoint --data $test_data --snp_mark_file $snp_mark_file --test_result_file $test_result_file --logger_file $logger_file

#done;
#done



