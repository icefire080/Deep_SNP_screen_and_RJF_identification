
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

# Reference for Model Architecture and Hyperparameter Selection

## Table S5a. Model comparison.

| Model | Core mechanism | Genetic signal captured (expected) | Pros | Cons / caveats | Interpretability / how to report | When it works best / failure modes | Data & compute needs | Key references |
|---|---|---|---|---|---|---|---|---|
| MLP (Multilayer Perceptron) | Feedforward network with fully-connected layers + nonlinear activations. | Nonlinear effects and higher-order interactions (implicit). | Flexible decision boundary; can capture complex multi-locus patterns. | Overfitting risk; sensitive to architecture/regularization; harder to interpret. | Moderate: needs post-hoc explanations (e.g., SHAP/permutation); not a direct map to causality. | Best with sufficient n and regularization; may fail with small n/high p or confounding stratification leakage. | Medium compute; benefits from GPU; higher data needs than LR. | Sheehan & Song 2016; Schrider & Kern 2018; Lundberg & Lee 2017 |
| DCN (Deep & Cross Network) | Cross layers explicitly construct bounded-degree feature interactions + deep network. | Low–moderate order SNP interactions plus nonlinear patterns. | Parameter-efficient interaction modeling; often stable for interaction signals. | Interaction order limited by #cross layers; interpretability still limited vs LR. | Moderate: structured interactions, but linking to biological epistasis requires care. | Best when modest-order SNP combinations matter; may fail for very high-order dependencies or too-small training sets. | Medium compute; tune cross-depth. | Wang et al. 2017; Schrider & Kern 2018 (context) |
| DFM / DeepFM | FM models pairwise interactions; deep part models higher-order nonlinear patterns end-to-end. | Pairwise + higher-order interactions among SNPs. | Balances efficiency (pairwise) and expressivity (deep); reduces manual interaction features. | More components to tune; can overfit without regularization; limited interpretability. | Moderate: pairwise part is closer to interaction intuition; global explanations usually post-hoc (e.g., SHAP). | Best when both low- and high-order interactions contribute; may fail with confounding, limited n, or poorly tuned embeddings. | Medium–high compute depending on embedding/depth; GPU helpful. | Guo et al. 2017; Schrider & Kern 2018; Lundberg & Lee 2017 |
| SA (Self-attention) | Attention adaptively re-weights/aggregates features; captures dependencies across features. | Distributed polygenic patterns; long-range dependencies. | Adaptive feature weighting; can highlight informative SNP sets; good for distributed signals. | Compute overhead; attention maps are heuristic; may be unstable with small n/noisy genotypes. | Potential: attention maps + SHAP-like tools; interpret as importance heuristics, not causal loci. | Best when signal is spread across many SNPs; may fail by overfitting confounding structure if n is small. | High compute relative to LR; careful regularization; GPU recommended. | Vaswani et al. 2017; Sheehan & Song 2016 (context); Lundberg & Lee 2017 |
| DFM-SA-MLP (fusion model; this study) | SNPs are encoded as tokens → embedding. Three parallel branches are computed: (1) DeepFM/DFM captures pairwise (FM) + higher-order (deep) interactions; (2) self-attention learns context-dependent feature weighting/dependencies (Q/K/V attention); (3) MLP learns general nonlinear decision boundaries. Branch outputs are fused by summation (or weighted fusion) and mapped to class probabilities via Softmax. | Additive + interaction-driven signals (pairwise to higher-order epistasis-like patterns) and distributed polygenic patterns where many SNPs collectively contribute; can also leverage long-range dependencies across loci in the learned representation. | - | - | - | - | Medium–high compute; benefits from GPU; usually needs more data than LR for stable generalization; strong regularization + early stopping recommended. | DeepFM (Guo et al., 2017); Self-attention/Transformer (Vaswani et al., 2017); DL in population genetics context (Sheehan & Song, 2016; Schrider & Kern, 2018); model-interpretation tools (Lundberg & Lee, 2017) |

## Table S5b. Hyperparameters and tuning guide

| Hyperparameter (plain name) | Value used in this study | What it does (plain language) | If set too high / too low, what you typically see | A simple tuning recipe (what to try first) | Notes / citations |
|---|---|---|---|---|---|
| Batch size | 32 | How many individuals are processed before one parameter update. | Too large: poorer generalization / memory issues; too small: noisy/slow training. | Try 16, 32, 64; choose largest that fits memory without hurting validation. | Config value. |
| Epochs | 15 (with early stopping) | Maximum passes through training data. | Too low: underfitting; too high without early stopping: overfitting. | Set upper bound and rely on early stopping; 15 was sufficient here. | Early stopping controls effective epochs. |
| Learning rate (lr) | 0.001 | How big each training step is. | Too high: divergence/oscillation; too low: very slow, may underfit. | Tune first on a log grid: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2; pick best validation metric with stable training. | Often the most sensitive setting. |
| Optimizer | AdamW | Update rule; AdamW is a stable default for deep models. | SGD often needs careful LR/momentum tuning. | Start with AdamW; only switch if needed. | AdamW: Loshchilov & Hutter 2019. |
| Weight decay (optimizer) | 0.01 | Mild shrinkage applied during optimization to reduce overfitting (applies to all parameters). | Too high: underfit; too low: overfit (train high, val drops). | After lr, tune 0, 1e-4, 1e-3, 1e-2, 5e-2; pick best validation. | Decoupled in AdamW. |
| Embedding regularization (regular_weight) | 0.0001 | Extra penalty on SNP embedding table 'emb.weight': ∑_rows ||w_i||₂ (L2,1 row-norm sum), encouraging group-wise sparsity (some rows shrink). | Too high: ignores useful SNP embeddings; too low: embeddings may overfit noise. | Tune 1e-5~1e-3 on log grid. If the model has no 'emb.weight', this term is skipped. | Group-lasso idea: Yuan & Lin 2006. |
| Loss function | Focal loss | Training objective; focal loss helps when classes are imbalanced by focusing on hard examples. | If imbalance is mild, may not help; if γ too large, training can be unstable. | Start with focal loss under imbalance; compare to (weighted) cross-entropy baseline. | Focal loss: Lin et al. 2017. |
| Focal loss α (alpha) | 0.9 | Up-weights the minority class in the loss. | Too high: more false positives; too low: misses minority class (low recall). | Try 0.5, 0.7, 0.9, 0.95; choose based on conservation goal (recall vs precision). | Controls sensitivity vs specificity. |
| Focal loss γ (gamma) | 1.5 | How strongly the loss focuses on misclassified (hard) samples. | Too high: noisy/unstable; too low: similar to cross-entropy. | Try 0, 1, 1.5, 2, 3; reduce if unstable. | Controls focusing strength. |
| Oversampling ratio | 1.5 | Repeats minority-class individuals during training to reduce imbalance. | Too high: overfit duplicates; too low: minority underrepresented. | Try 1.0, 1.5, 2.0; choose by validation F1/recall. | Complements focal loss. |
| Learning-rate scheduler | ReduceLROnPlateau (type=plateau) | Reduces LR when validation metric stops improving. | Too aggressive: stalls early; too mild: prolonged plateau. | Use plateau when unsure of schedule; start with factor 0.5 and patience 2~5. | PyTorch ReduceLROnPlateau. |
| Plateau monitor / mode | monitor validation F1, mode="max" | Tells the scheduler what to watch and whether higher is better. | Monitoring the wrong metric triggers LR drops at wrong times. | Choose one primary metric (F1 for imbalance; AUC for ranking) and monitor it consistently. | Config comment indicates F1. |
| Scheduler factor / patience / min_lr | 0.5 / 3 / 1e-6 | Controls how much/when LR is reduced and the minimum LR. | Very small min_lr can stall; too large factor can be abrupt. | If LR drops too often, increase patience; if never drops, decrease patience. | Config value. |
| Early stopping patience | 5 | Stops training if validation does not improve for this many epochs. | Too small: stops early; too large: overfits/wastes compute. | Start 5~10; if validation is noisy, increase a bit. | Prechelt 1998. |
| Early stopping delta | 0.001 | Minimum improvement needed to count as progress. | Too large: premature stop; too small: trains on noise. | Start 0.001~0.01 (for F1 scale); 0.005 is practical. | Config value. |

## References

- [Schrider & Kern 2018] Schrider DR, Kern AD. Supervised Machine Learning for Population Genetics: A New Paradigm. Trends Genet. 2018.
- [Sheehan & Song 2016] Sheehan S, Song YS. Deep Learning for Population Genetic Inference. PLoS Comput Biol. 2016;12(3):e1004845.
- [Loshchilov & Hutter 2019] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization (AdamW). ICLR. 2019.
- [Lin et al. 2017] Lin T-Y, Goyal P, Girshick R, He K, Dollár P. Focal Loss for Dense Object Detection. ICCV. 2017.
- [Prechelt 1998] Prechelt L. Automatic early stopping using cross-validation: quantifying the criteria. Neural Networks. 1998;11(4):761–767.
- [Yuan & Lin 2006] Yuan M, Lin Y. Model Selection and Estimation in Regression with Grouped Variables. JRSSB. 2006;68(1):49–67.


### Contributors
Primary Developer: Bei Liu (bei_liu_go@163.com)

Contributor: Zhengfei Cai (caizhengfei686@qq.com)

Supervisor: MS. Peng
