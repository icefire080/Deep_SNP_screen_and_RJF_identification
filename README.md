
# Deep_SNP_Screen_and_RJF_identification （DeepSNP&RJF-ID）


DeepSNP&RJF-ID is a tool for molecular identification of Red Jungle Fowl (RJF). Sample the genotype of the chicken to be identified on the specified SNPs (~700) and input it into the AI ​​model in the tool to get the identification result of whether it is a red jungle fowl.

In addition to providing the trained identification model, the tool also provides the entire training process code. The whole process includes

1. Training data preparation: parse the sample VCF file and join label information.
2. SNP fine screening: Use computer perturbation experiments to screen SNPs with high information content and high contribution to identification.
3. Identification model training: train a new model from scratch
4. Qualification model testing: Use the interface of the qualification model.

In addition to the molecular identification of red junglefowl, the tool can also be used for the molecular identification of other species.


<p align="center">

<img src="https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification/blob/main/docs/Figure1.png" width="500" height="900">

</p>


### Step 1: Clone Repository
```bash
git clone https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification.git
cd Deep_SNP_screen_and_RJF_identification
```

### Step 2: Create Conda Environment
```bash
conda create --name deepsnp_env python=3.10 -y
conda activate deepsnp_env
pip install -r requirements.txt
```

### Quick Start: Sample Identification (such as RJF)

Step 1: Prepare Genotype File
```bash
vcftools --gzvcf your.vcf.gz \
         --extract-FORMAT GT \
         --positions identification_model/RJF/285-SNP.list.txt \
         --recode \
         --out your.GT
mv your.GT.recode.vcf your.GT.vcf
```

Step 2: Convert to Pickle Format
```bash
python script_v2.0/rjf_sample_feat_parse.py \
  --method parse_vcf \
  --vcf_file your.GT.vcf \
  --data_file your.GT.pickle \
  --mock_group 1
```

Step 3: Run Identification
```bash
python script_v2.0/rdc_classify_test.py \
  --checkpoint identification_model/RJF/model_checkpoint \
  --logger_file identification.log \
  --data your.GT.pickle \
  --test_result_file prediction_results.txt \
  --mock_y 1
```

### Train Custom Model
Step 1: Prepare Training Data
```bash
# Convert VCF to pickle format
python script_v2.0/rjf_sample_feat_parse.py \
  --method parse_vcf \
  --vcf_file your_full.vcf \
  --sample_2_group_file breed_labels.txt \
  --data_file training_data.pickle
```

```
# Split dataset (80% train, 20% test)
python script_v2.0/rjf_sample_feat_parse.py \
  --method split_train_test \
  --data_file training_data.pickle
```

Step 2: SNP Panel Optimization
```bash
# Create config file
echo "data:
 train_data : rawdataset/3.7k_snp.pickle.train # input pickle file
 mit_flag : 0 # for mitochondrial lables, keep it as 0
 stop_snps : ./dcn/search_stop_snp.rm_info.epoch10 # epoch for stoping list
 snp_select_model_filter_breed : ggs #in case reporting an error, please keep it

train_param:
 epoch : 64
 lr : 0.001 #learning rate
 batch_size : 5
 model_flag : Dcn # options for architectures: Snp_dnn_simple, Snp_dnn_lr, Dfm, main, Snp_transform, Dcn, Snp_dnn_2_layer, Snp_dnn_4_layer
 regular_weight : 0.0
 cuda : cuda:1 # GPU

snp_search_train_param:
 epoch : 30
 lr : 0.005
 batch_size : 64
 model_flag : Dcn 
 regular_weight : 0.0
 cuda : cuda:1
 test_flag : false
 cpu : False

snp_search:
 mask_diff_base_model_auc_th : 0.88 #
 max_find_epoch : 62 #max find epoch
 mask_diff_th : 0.96 #difference threshold
 find_step : 100 # steps for SNP removing 
 target_snp_cnt : 300 # target SNP panel size
 result_remove_snp : ./dcn/search_stop_snp #stoping list of each steps
 test_flag : false " > snp_config.yml
```

```
# Run screening
mkdir snp_selection
python script_v2.0/mask_2.py \
  --method mask_weight \
  --conf snp_config.yml
```

Step 3: Train Identification Model
```bash
# Create training config
echo "data:
  batch_sampler: none # keep it
  filter_breed: ggs # keep it
  init_stop_snps: ./none # keep it
  mit_flag: 0 
  oversample_ratio: 1.5 
  stop_snps: ./data/search_stop_snp.epoch33 #stoping list for 285-SNP panel
  train_data: ./data/3.7k_snp.pickle.train #traning data
train_param:
  cuda: cuda:0
  model_flag: Dcn
  batch_size: 32
  epoch: 15
  lr: 0.001
  weight_decay: 0.01
  regular_weight: 0.0001
  print_global_step: 60  
  early_stopping_patience: 5  
  early_stopping_delta: 0.005
  save_dir: ./result/model_checkpoint # path to model
  test_flag: false
  
  optimizer: "adamw"  # options: adamw, adam, sgd
  momentum: 0.9  # for SGD
  
  loss_fn: "focal"  # options: cross_entropy, focal
  label_smoothing: 0.1  # for CrossEntropyLoss
  focal_alpha: 0.9  # for FocalLoss
  focal_gamma: 1.5 # for FocalLoss
  
  scheduler:
    type: "plateau"  # options: step, cosine, plateau, onecycle
    factor: 0.5  
    patience: 3   
    min_lr: 1e-6  # minimum learning rate
    # for StepLR
    step_size: 10
    gamma: 0.1
    # for CosineAnnealingLR
    t_max: 50
    eta_min: 1e-5
    # for OneCycleLR
    max_lr: 0.01
    total_steps: 100
    pct_start: 0.3
    final_div_factor: 1000
    # for Plateau
    mode: "max"  # monitoring F1  
utils:
  checkpoint: ./result/model_checkpoint # path to model
  logger_file: ./result/process.log # log file
  logging_level: 11
  model_dir : ./result/ # the path for the final model" > train_config.yml
```

```
#Start training
mkdir trained_model
python script_v2.0/rdc_identify_model.py \
  --method train_with_stop_snp \
  --conf train_config.yml
```

Step 4: Validate Model
```bash
python script_v2.0/rdc_classify_test.py \
  --checkpoint ./trained_model/model_checkpoint \
  --data training_data.pickle.test \
  --snp_mark_file final_snp_panel.txt \
  --test_result_file validation_results.txt \
  --logger_file validation.log
```

### Important Notes
Label Requirements

For non-RJF species, maintain these labels in your breed info file:

```
SampleID    Group
Sample1     DC    # Domestic animals, represents negative samples
Sample2     RJF   # Wild relatives, represents positive samples
```

### Contributors
Primary Developer: Bei Liu (bei_liu_go@163.com)

Contributor: Zhengfei Cai (caizhengfei686@qq.com)

Supervisor: MS. Peng
