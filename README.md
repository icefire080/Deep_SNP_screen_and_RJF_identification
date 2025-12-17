
# Deep_SNP_Screen_and_RJF_identification （DeepSNP&RJF-ID）

DeepSNP&RJF-ID is a tool designed for the molecular identification of Red Jungle Fowl (RJF). It first samples the chicken’s genotype at approximately 300 specified SNPs and then analyzes this data through an AI model within the tool to determine whether the chicken is a true Red Jungle Fowl.

In addition to providing a trained identification model, DeepSNP&RJF-ID also includes the complete training workflow, which comprises the following steps:

1. Training Data Preparation: Parsing sample VCF files and attaching corresponding label information.
2. SNP Refinement: Employing computational perturbation experiments to identify SNPs with high information content and strong discriminatory power.
3. Model Training: Training a new identification model from scratch.
4. Model Qualification and Testing: Using the built-in interface to validate and evaluate the trained model.

Although designed specifically for Red Jungle Fowl identification, DeepSNP&RJF-ID can also be adapted for the molecular identification of other species.


<p align="center">

<img src="https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification/blob/main/docs/Figure1.png" width="900" height="1600">

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
echo "utils: 
 logging_level : INFO
 model_dir : ./
 logger_file : ./process.log
 test_flag : false
 checkpoint : ./model_checkpoint
data:
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
 cuda : cuda:0 # Set up the GPU

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

### Data Information

```
3.7k_snp.vcf.gz: Raw VCF file used for training the Red Jungle Fowl (RJF) identification model.

5k.snp.Asian_only.new.vcf.gz: Raw VCF file used for training the wild boar identification model.

Breeds_and_group.information_chickens.txt: Label information for Red Jungle Fowl and domestic chicken breeds.

Breed_info_Asian_pigs_only.txt: Label information for wild boar and Asian domestic pig breeds.
```

# Details of architectures and hyperparameters

## Table S1A. Model comparison (MLP/DCN/DeepFM/Self-attention)

| Model | Core mechanism | Genetic signal captured (expected) | Strengths | Limitations | Key architecture hyperparameters (typical) | Key reference |
|---|---|---|---|---|---|---|
| MLP (feedforward neural network) | Stacked fully-connected layers with nonlinear activations. | Nonlinear SNP effects and higher-order interactions (implicit). | Flexible decision boundaries; can approximate complex genotype→class mappings. | May overfit; sensitive to architecture, regularization, and sample size. | Depth (#layers), width (#neurons), activation, dropout, batch norm. | Goodfellow et al. 2016 |
| DCN (Deep & Cross Network) | Parallel ‘cross’ layers explicitly construct bounded-degree feature crosses; combined with deep network. | Efficient low-to-moderate order feature interactions (explicit crossing). | Captures interactions with fewer parameters than very deep MLPs; often stable. | Interaction order limited by #cross layers; may miss very complex interactions. | #cross layers, deep depth/width, embedding/hidden size, dropout. | Wang et al. 2017 (DCN) |
| DFM / DeepFM | Factorization Machine part models pairwise interactions; deep part models higher-order nonlinear patterns; trained end-to-end. | Both low-order (pairwise) and higher-order interactions among SNPs. | Balances efficiency (FM) and expressivity (deep); less manual feature engineering. | More components to tune; interpretability limited compared with LR. | Embedding dimension (FM), deep depth/width, dropout, activation. | Guo et al. 2017 (DeepFM) |
| SA (Self-attention) | Computes attention weights to re-weight and combine feature representations; captures dependencies across input positions/features. | Distributed signals; long-range dependencies; context-dependent feature importance. | Adaptive weighting; can provide attention scores as heuristic importance. | More compute; attention patterns can be hard to interpret biologically without care. | #heads, attention/hidden dim, #layers, dropout, positional encoding choice. | Vaswani et al. 2017 |

## Table S1B. Training and optimization hyperparameters

| Item | Value used in this study | What it is | Why it matters | Typical range / notes |
|---|---:|---|---|---|
| Batch size | 32 | Individuals per gradient update. | Affects gradient noise, speed, generalization. | 16–128 |
| Max epochs | 15 | Upper bound on training passes through data. | Controlled with early stopping to avoid overfitting. | 10–200 |
| Optimizer | AdamW | Adaptive optimizer with decoupled weight decay. | Often improves generalization vs. Adam+L2 coupling. | Adam / AdamW / SGD |
| Learning rate | 0.001 | Step size of parameter updates. | Most sensitive; too high diverges, too low underfits. | 1e-4–1e-2 (log-scale) |
| Weight decay | 0.01 | Decoupled shrinkage on weights (AdamW). | Regularizes and stabilizes training. | 0–0.05 |
| L2 regularization weight | 0.0001 | Additional penalty term (if used in loss). | Controls parameter magnitude; reduces overfitting. | 0–1e-3 |
| LR scheduler | ReduceLROnPlateau | Reduces LR when validation metric plateaus. | Helps refine convergence after stagnation. | Factor 0.1–0.8; patience 2–10 |
| Scheduler factor | 0.5 | Multiply LR by factor when plateau detected. | Smaller factor = stronger LR drop. | 0.1–0.8 |
| Scheduler patience | 3 | Epochs without improvement before reducing LR. | Prevents premature LR drops. | 2–10 |
| Minimum LR | 1e-6 | Lower bound for LR after reductions. | Avoids LR becoming too small. | 1e-8–1e-5 |
| Early stopping delta | 0.001 | Minimum improvement to be considered progress. | Controls sensitivity to small metric changes. | 1e-4–1e-2 |
| Early stopping patience | 5 | Epochs to wait before stopping if no progress. | Stops before overfitting; improves reproducibility. | 3–20 |
| Loss | Focal loss | Re-weights easy vs. hard examples; addresses imbalance. | Useful under class imbalance; focuses hard samples. | CE / weighted CE / focal |
| Focal loss α | 0.9 | Class weighting (up-weights minority class). | Higher α increases minority-class emphasis. | 0.5–0.95 |
| Focal loss γ | 1.5 | Focusing parameter (down-weights easy examples). | Higher γ increases focus on hard samples. | 0–3 |
| Repeated runs | 5 | Independent training runs with different random seeds. | Quantifies training variability. | 3–10 |
| Missingness stress test | 5%–50% (step 5%), 20 repeats | Randomly mask SNPs to simulate missing genotypes. | Evaluates robustness to incomplete genotyping. | Study-dependent |

## References

- **Goodfellow et al. 2016**: Goodfellow I, Bengio Y, Courville A. Deep Learning. MIT Press; 2016.
- **Wang et al. 2017 (DCN)**: Wang R, Fu B, Fu G, Wang M. Deep & Cross Network for Ad Click Predictions. ADKDD@KDD; 2017.
- **Guo et al. 2017 (DeepFM)**: Guo H, Tang R, Ye Y, Li Z, He X. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. IJCAI; 2017.
- **Vaswani et al. 2017**: Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS; 2017.
- **Loshchilov & Hutter 2019**: Loshchilov I, Hutter F. Decoupled Weight Decay Regularization (AdamW). ICLR; 2019.
- **Lin et al. 2017**: Lin T-Y, Goyal P, Girshick R, He K, Dollár P. Focal Loss for Dense Object Detection. ICCV; 2017.
- **Prechelt 1998**: Prechelt L. Automatic early stopping using cross-validation: quantifying the criteria. Neural Networks. 1998;11(4):761–767.
- **PyTorch ReduceLROnPlateau**: PyTorch documentation: torch.optim.lr_scheduler.ReduceLROnPlateau (accessed 2025).

### Contributors
Primary Developer: Bei Liu (bei_liu_go@163.com)

Contributor: Zhengfei Cai (caizhengfei686@qq.com)

Supervisor: MS. Peng
