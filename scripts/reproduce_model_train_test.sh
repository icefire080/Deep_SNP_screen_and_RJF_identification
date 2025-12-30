
source activate pytorch_v1
echo "change to conda in gpu server"
echo "----begin identify moddel train------"

exp_dir=exp_demo


## for train 
config_file=$exp_dir/config_test.yml

## for test 
test_data=$exp_dir/GGS_VS_DC_demo.pickle.test
logger_file=$exp_dir/test.log
snp_mark_file=$exp_dir/snp_tokens
test_result_file=$exp_dir/test_result
checkpoint=$exp_dir/model_checkpoint


echo "--begin to trian model"

python script_v2.0/rdc_identify_model.py  --method train_with_stop_snp  --conf $config_file 

echo "----begin identify moddel test------"

python script_v2.0/rdc_classify_test.py --checkpoint $checkpoint --data $test_data --snp_mark_file $snp_mark_file --test_result_file $test_result_file --logger_file $logger_file 
