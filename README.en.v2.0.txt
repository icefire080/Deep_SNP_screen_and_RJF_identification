# DeepSNP&RJF Identification Tool

DeepSNP&RJF-ID is an AI-powered tool for molecular identification of Red Jungle Fowl (RJF) and other species. By genotyping samples at specific SNPs (~700 sites) and processing through our AI model, it determines whether a sample belongs to the red jungle fowl lineage.

The tool provides:
- Pre-trained identification models
- Full training pipeline code
- Customizable workflow for other species

<p align="center">
  <img src="https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification/blob/main/docs/main_1.png" width="900" height="280">
</p>

---

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification.git
cd Deep_SNP_screen_and_RJF_identification
```

Step 2: Create Conda Environment
```bash
conda create --name deepsnp_env python=3.10 -y
conda activate deepsnp_env
pip install -r requirements.txt
```

🚀 Quick Start: Sample Identification
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

🔧 Train Custom Model
Step 1: Prepare Training Data
```bash
# Convert VCF to pickle format
python script_v2.0/rjf_sample_feat_parse_new.py \
  --method parse_vcf \
  --vcf_file your_full.vcf \
  --sample_2_group_file breed_labels.txt \
  --data_file training_data.pickle
```

```
# Split dataset (80% train, 20% test)
python script_v2.0/rjf_sample_feat_parse_new.py \
  --method split_train_test \
  --data_file training_data.pickle
```

Step 2: SNP Panel Optimization
```bash
# Create config file
echo "utils:
  logging_level: INFO
  model_dir: ./snp_selection/
  logger_file: ./snp_selection/process.log
data:
  train_data: training_data.pickle.train
  mit_flag: 0
train_param:
  cuda: cuda:0
snp_search:
  target_snp_cnt: 300
  result_remove_snp: ./snp_selection/selected_snps.txt" > snp_config.yml

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
  stop_snps: ./snp_selection/selected_snps.txt
  train_data: training_data.pickle.train
train_param:
  cuda: cuda:0
  model_flag: Dcn
  epoch: 15
  save_dir: ./trained_model/
utils:
  checkpoint: ./trained_model/model_checkpoint
  logger_file: ./trained_model/training.log" > train_config.yml
```

```
# Start training
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

📝 Important Notes
Label Requirements
For non-RJF species, maintain these labels in your breed info file:

plaintext
SampleID    Group
Sample1     DC    # Domestic animals
Sample2     RJF   # Wild relatives
File Paths
Always use absolute paths in configuration files:

yaml
train_data: /home/user/project/data/training.pickle.train
Hardware Requirements


👥 Contributors
Primary Developer: Liu Bei

Contributor: Cai Zhengfei

Supervisor: MS. Peng