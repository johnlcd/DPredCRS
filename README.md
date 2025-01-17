# DPredCRS
> Predicting DVT Occurrence Based on ***`Clinical Risk Stratification`***
<br>

## Introduction

> Deep venous thrombosis (DVT) is a prevalent complex disease that is influenced by various factors.  
> This tool, `DPredCRS`, is used for predicting DVT occurrence using clinical, genetic and metabolic features based on the stratication of clinical risk stratification.  
<br>

## License
> This software is distributed under the terms of GPL 3.0  
<br>

## Source
> [https://github.com/johnlcd/DPredCRS](https://github.com/johnlcd/DPredCRS)  
<br>

## Usage
> usage: RFE_Train.py [-h] [--cpu {0,1}] [--fs {0,1}] [--seed SEED] --dataset DATASET --rapt {5,6,7,8,9,10,11,12,13,14} [--func {rfecv,rfe}]
                    [--rfe_classifier {RF,LR,LSVM}] [--classifier {RF,NB,KNN,LR,DT,LSVM,SVMR,SVMP,GBDT}] [--scorer {roc_auc,f1}] [--nest NEST]
                    [--mdepth MDEPTH] [--min_fs MIN_FS] [--step STEP] [--nf_select NF_SELECT] [--model_save_file MODEL_SAVE_FILE] --dpath DPATH --rpath RPATH

optional arguments:
  -h, --help            show this help message and exit
  --cpu {0,1}           Device Used (default: [ "0" ] (CUDA), Option: [ "1" ] (CPU))
  --fs {0,1}            Feature Selection mode (default: [ "1" ] (Run Feature Selection), Option: [ "0" ] (Loading Saved Models))
  --seed SEED           Random Seed (default: [ "0" ] (Random))
  --dataset DATASET     Dataset Prefix
  --rapt {5,6,7,8,9,10,11,12,13,14}
                        RAPT Score for Grouping
  --func {rfecv,rfe}    Function of Feature Selection (default: [ "rfecv" ])
  --rfe_classifier {RF,LR,LSVM}
                        Feature selection Classifier (default: [ "RF" ] (RandomForest Classifier))
  --classifier {RF,NB,KNN,LR,DT,LSVM,SVMR,SVMP,GBDT}
                        Machine Learning Classifier (default: [ "RF" ] (RandomForest Classifier))
  --scorer {roc_auc,f1}
                        Scorer of Model Performace (default: [ "roc_auc" ])
  --nest NEST           Number of Estimators (Trees) (defult: [ 100 ], "--rfe_classifier RF")
  --mdepth MDEPTH       Max Depth of Tree (defult: [ 2 ], "--rfe_classifier RF")
  --min_fs MIN_FS       Minimum Number of Features to be Selected (defult: [ 1 ])
  --step STEP           Step Size of Feature Selection (defult: [ 1 ])
  --nf_select NF_SELECT
                        Number of Features to Select (defult: [ 10 ], "--func rfe")
  --model_save_file MODEL_SAVE_FILE
                        File to Save Machine Learning Model (defult: [ "None" ])
  --dpath DPATH         Data Path for Feature Matrix
  --rpath RPATH         Result Directory



