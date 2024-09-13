#!/usr/bin/env python

import os,sys
import argparse
import torch
import numpy as np
import warnings
import pickle as pkl
import dpctl
from sklearnex import patch_sklearn, config_context

#import utils_rfe
from utils_rfe import load_data_gp, DataSet, seed_everything
from models_rfe import ML


warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cpu', type=int, default=0, choices=[0,1], help='Device Used (default: [ \"0\" ] (CUDA), Option: [ \"1\" ] (CPU))')
parser.add_argument('--fs', type=int, default=1, choices=[0,1], help='Feature Selection mode (default: [ \"1\" ] (Run Feature Selection), Option: [ \"0\" ] (Loading Saved Models)) ')
parser.add_argument('--seed', type=int, default=0, help='Random Seed (default: [ \"0\" ] (Random))')
parser.add_argument('--dataset', type=str, default='Clinical_Meta_Geno', help='Dataset Prefix', required=True)
parser.add_argument('--rapt', type=str, default='10', choices=['5','6','7','8','9','10','11','12','13','14'], help='RAPT Score for Grouping', required=True)
parser.add_argument('--func', type=str, default='rfecv', choices=['rfecv','rfe'], help='Function of Feature Selection (default: [ \"rfecv\" ])')
parser.add_argument('--rfe_classifier', type=str, default='RF', choices=['RF','LR','LSVM'], help='Feature selection Classifier (default: [ \"RF\" ] (RandomForest Classifier))')
parser.add_argument('--classifier', type=str, default='RF', choices=['RF','NB','KNN','LR','DT','LSVM','SVM','GBDT'], help='Machine Learning Classifier (default: [ \"RF\" ] (RandomForest Classifier))')
parser.add_argument('--scorer', type=str, default='roc_auc', choices=['roc_auc','f1'], help='Scorer of Model Performace (default: [ \"roc_auc\" ])')
parser.add_argument('--nest', type=int, default=100, help='Number of Estimators (Trees) (defult: [ 100 ], \"--rfe_classifier RF\")')
parser.add_argument('--mdepth', type=int, default=2, help='Max Depth of Tree (defult: [ 2 ], \"--rfe_classifier RF\")')
parser.add_argument('--min_fs', type=int, default=1, help='Minimum Number of Features to be Selected (defult: [ 1 ])')
parser.add_argument('--step', type=float, default=1, help='Step Size of Feature Selection (defult: [ 1 ])')
parser.add_argument('--nf_select', type=int, default=10, help='Number of Features to Select (defult: [ 10 ], \"--func rfe\")')
parser.add_argument('--model_save_file', type=str, default="None", help='File to Save Machine Learning Model (defult: [ \"None\" ])')

args = parser.parse_args()
device = torch.device("cuda" if (torch.cuda.is_available() and args.cpu != 1) else "cpu")
args.device = device
print("\n  JOB OVERVIEW:")
print("\n>>> Machine Learning Framework Using Device:     [ \'" + device.type + "\' ]\n")
print("##### All ARGS of the program:\n",args)

#seed_everything(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True

## load dataset
print("\n\n\n===============================================  Loading DataSet  ================================================")
print("\n##### DataSet:     [ \"{}\" ] ".format(args.dataset))
print("\n##### RAPT score for grouping:     [ \"{}\" ] ".format(args.rapt))
data_path = "/home/chenjiabin/project/DVT/data/All_comb/model_pred/data"
data0 = load_data_gp(data_path, args.dataset, args.rapt, 0, args.seed).to(device)
data1 = load_data_gp(data_path, args.dataset, args.rapt, 1, args.seed).to(device)
data2 = load_data_gp(data_path, args.dataset, args.rapt, 2, args.seed).to(device)
print("\n======================================================================================================================\n")

model = ML(args)

## Feature selection, Training and Evaluation
### feature selection
result_path = "/home/chenjiabin/project/DVT/data/All_comb/model_pred/results"
#model_path = "/home/chenjiabin/project/DVT/data/All_comb/model_pred/results/model_save"
model_save_f0 = "{}/model_save/{}_R{}_G0_{}_RFE.pkl".format(result_path, args.dataset, args.rapt, args.rfe_classifier)
model_save_f1 = "{}/model_save/{}_R{}_G1_{}_RFE.pkl".format(result_path, args.dataset, args.rapt, args.rfe_classifier)
model_save_f2 = "{}/model_save/{}_R{}_G2_{}_RFE.pkl".format(result_path, args.dataset, args.rapt, args.rfe_classifier)
if (args.fs!=1 and os.path.exists(model_save_f0) and os.path.exists(model_save_f1) and os.path.exists(model_save_f2)):
	print(">>> Loading saved model (RFE) \n... \n... ... \n... ... ...\n")
	print("### Model save file0:  \"{}\"".format(model_save_f0))
	print("### Model save file1:  \"{}\"".format(model_save_f1))
	print("### Model save file2:  \"{}\"".format(model_save_f2))
	model_rfe0,fea_rfe0,idx_rfe0 = pkl.load(open(model_save_f0, 'rb'))
	model_rfe1,fea_rfe1,idx_rfe1 = pkl.load(open(model_save_f1, 'rb'))
	model_rfe2,fea_rfe2,idx_rfe2 = pkl.load(open(model_save_f2, 'rb'))
else:
	model0 = model
	model1 = model
	model2 = model
	## Group0 (All)
	print("\n\n========================================  Feature selection of Group 0 (ALL)  ========================================")
	model_rfe0, fea_rfe0, idx_rfe0 = model0.rfe_cv(data0)
	print("\n======================================================================================================================\n")
	## Group1
	print("\n\n===========================================  Feature selection of Group 1  ===========================================")
	model_rfe1, fea_rfe1, idx_rfe1 = model1.rfe_cv(data1)
	print("\n======================================================================================================================\n")
	## Group2
	print("\n\n===========================================  Feature selection of Group 2  ===========================================")
	model_rfe2, fea_rfe2, idx_rfe2 = model2.rfe_cv(data2)
	print("\n======================================================================================================================\n")
### training && evaluation
print("\n\n====================================  Training and evaluation of combined  model  ====================================")
model.fit(data0, data1, data2, model_rfe0, model_rfe1, model_rfe2, fea_rfe0, fea_rfe1, fea_rfe2)
#model.fit(data1, data2, model_rfe1, model_rfe2, fea_rfe1, fea_rfe2, idx_rfe1, idx_rfe2)
print("\n======================================================================================================================\n\n")


