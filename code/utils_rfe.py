import torch
import random
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn import preprocessing
import pickle as pkl
import os
import sys
import networkx as nx
from collections import Counter


def seed_everything(seed=0):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_index_f(path):
	"""Parse the index file.
	Parameters
	----------
	path:
		directory of index file (str)
	"""
	index = []
	for line in open(path):
		index.append(int(line.strip()))
	return index


def get_mask(idx, l):
	"""Create mask.
	"""
	mask = torch.zeros(l, dtype=torch.bool)
	mask[idx] = 1
	return mask


def normalize(mx):
	"""Row-normalize sparse matrix.
	"""
	r_sum = np.array(mx.sum(1))
	r_inv = np.power(r_sum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def fea_norm(mx, p = 1, t = False):
	"""Normalize features by standard division of each feature (t = False ==> "normalized by feature" )
	"""
	print("\n##### Features matrix (sparse matrix) [ {} , {} ] ==>".format(np.array(mx).shape[0],np.array(mx).shape[1]))
	print("### Raw:")
	print(mx)
	mx = mx.astype(np.float32)
	## normalization by each sample
	if t == False:
		mx = mx.T
	# calculate variance
	r_std = mx.std(axis = 1)
	r_inv = np.power(r_std, -1/p).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx_norm = r_mat_inv.dot(mx)
	if t == False:
		mx_norm = mx_norm.T
	#mx_norm = sklearn.preprocessing.scale(mx, axis=1, with_mean=False, with_std=True, copy=True)
	print("### Normalized:")
	print(mx_norm)
	return mx_norm

def scale_norm_fea(mx, with_mean = True, with_std = True, norm = 'l2'):
	"""Standardization and Normalization (mean, standard division) of each feature
	"""
	## Standardization
	print("\n>>> Standardization and Normalization of feature \n... \n... ... \n... ... ...")
	print("### Raw feature matrix:\n   ",mx)
	print("=========================   Standardization (Mean, Standard Division)   =========================")
	scaler = preprocessing.StandardScaler().fit(mx)
	mx_scale = scaler.transform(mx)
	print("### Standardization (transformed) feature matrix:\n   ",mx_scale)
	print("=================================================================================================")
	print("=======================================   Normalization   =======================================")
	normalizer = preprocessing.Normalizer(norm).fit(mx_scale)
	mx_scale_norm = scaler.transform(mx_scale)
	print("### Normalization (transformed) feature matrix:\n   ",mx_scale_norm)
	print("=================================================================================================")
	return mx_scale, mx_scale_norm

def idx_samp(label, rt1, rt2, seed):
	"""\"rt1\": ratio of train_all/all data (train_all = train + val); \"rt2\": ratio of train/val data;
	"""
#ntest": "0" for all rest data; "rand": "0" for not random, other for random; "seed": "0" for not seed
	nsamp = len(label)
	ntrain_all = round(nsamp*float(rt1))
	ntrain = round(ntrain_all*float(rt2))
	nval = ntrain_all - ntrain
	ntest = nsamp - ntrain_all
	idx_all = range(0,nsamp,1)
	idx_train_all = []
	idx_train = []
	idx_val = []
	idx_test = []
	lb_class = set(label)
	nclass = len(lb_class)
	print("\n### Number of Sample: ")
	print("   ", nsamp)
	print('\n### Label of Class: ')
	print("    [ {} , {} ] ".format(str(list(lb_class)[0]), str(list(lb_class)[1])))
	for lb in lb_class:
		print('\n******************** Label \"{}\" ********************'.format(lb))
		idx_lb = []
		idx_lb.extend([i for i,x in enumerate(label) if x==lb])
		nlabel = len(idx_lb)
		ratio = float(nlabel/nsamp)
		print("### Ratio of sampling ")
		print("   ", ratio)
		ntrain_all_lb = round(ntrain_all*ratio)
		ntrain_lb = round(ntrain*ratio)
		nval_lb = ntrain_all_lb - ntrain_lb
		ntest_lb = round(ntest*ratio)
		if seed != 0:
			np.random.seed(seed)
		else:
			np.random.seed()
		np.random.shuffle(idx_lb)
		print("### Number of: [ Training (ALL) : Testing ]")
		print("    [ {} : {} ] ".format(ntrain_all_lb, ntest_lb))
		idx_train_all = idx_train_all + idx_lb[0:ntrain_all_lb]
		idx_train = idx_train + idx_lb[0:ntrain_lb]
		idx_val = idx_val + idx_lb[ntrain_lb:ntrain_all_lb]
		#idx_test = idx_test + idx_lb[ntrain_all_lb:nlabel]
		print('***************************************************')
	idx_test = list(set(idx_all) - set(idx_train_all))
	idx_train_all.sort()
	idx_train.sort()
	idx_val.sort()
	idx_test.sort()
	return idx_train_all,idx_train,idx_val,idx_test


# load data by group (independent)
def load_data_gp(path, dataset, rapt, group, seed):
	"""Load input data from directory.
	Parameters
	----------
	path:
		directory of data (str)
	dataset:
		name of dataset (str)
	rapt:
		RAPT score for grouping (str)
	group:
		group of dataset ("1" for high risk, "2" for medium and low risk)

	Files
	----------
	ind.dataset.x:
		feature of trainset (sp.csr.csr_matrix)
	ind.dataset.tx:
		feature of testset (sp.csr.csr_matrix)
	ind.dataset.allx:
		feature of both labeled and unlabeled training instances (sp.csr.csr_matrix)
	ind.dataset.y:
		one-hot label of trainset (numpy.array)
	ind.dataset.ty:
		one-hot label of testset (numpy.array)
	ind.dataset.ally:
		label of instances in ind.dataset.allx (numpy.array)
	ind.dataset.test.index:
		indices of testset for the inductive setting (list)

	All objects above must be saved using python pickle module.
	"""
	group=str(group)
	print("\n\n=====> Loading Group {} \n... \n... ... \n... ... ...\n###  Data path:  \" {} \" ".format(group,path))
	feature_dict = {}
	label_dict = {}
	file_fea_label = "{}/RAPT_{}/{}_G{}_{}".format(path, rapt, dataset, group, '831_filtered_fea_label')
	file_fea_type = "{}/{}_fea_type".format(path, dataset)

	with open(file_fea_label, 'rb') as f:
		f.readline()
		for line in f.readlines():
			line = line.decode().rstrip().split('\t')
			feature_dict[line[0]] = np.array(line[1].split(','), dtype=np.float64) #np.float_
			label_dict[line[0]] = int(line[2])

	with open(file_fea_type, 'r') as f:
		Feature_Type = {}
		Feature_Idx = {}
		idx = 0
		for line in f.readlines():
			Fea = line.strip().split('\t')[0]
			Type = line.strip().split('\t')[1]
			Feature_Type[Fea] = Type
			Feature_Idx[Fea] = idx
			idx += 1

	fea_raw = np.array(list(feature_dict.values()),dtype=np.float64)
	## Standardization
	fea_scale, fea_scale_norm = scale_norm_fea(fea_raw)
	feature = torch.from_numpy(fea_scale_norm).float()
	nfea_all = feature.shape[1]
	print("\n### Number of Feature: \n    [ {} ]".format(str(nfea_all)))
	label = np.array(list(label_dict.values()),dtype='i4')
	num_class = len(set(label))
	num_samp = len(label)
	idx_train_all,idx_train,idx_val,idx_test = idx_samp(label, 0.75, 0.8, seed)
	label = torch.LongTensor(label)

	mask_train_all = get_mask(idx_train_all, label.size(0)) #tensor of booler value for train data index (length of all label)
	mask_train = get_mask(idx_train, label.size(0)) #tensor of booler value for train data index (length of all label)
	mask_val = get_mask(idx_val, label.size(0))
	mask_test = get_mask(idx_test, label.size(0))
	return DataSet(dataset=dataset, rapt=rapt, group=group, x=feature, y=label, feature_type = Feature_Type, feature_idx = Feature_Idx, 
			idx_train_all=idx_train_all, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test, 
			mask_train_all=mask_train_all, mask_train=mask_train, mask_val=mask_val, mask_test=mask_test)


class DataSet():
	def __init__(self, dataset, rapt, group, x, y, feature_type, feature_idx, idx_train_all, idx_train, idx_val, idx_test, mask_train_all, mask_train, mask_val, mask_test):
		self.dataset = dataset
		self.rapt = rapt
		self.group = group
		self.x = x
		self.y = y
		self.feature_type = feature_type
		self.feature_idx = feature_idx
		self.idx_train_all = idx_train_all
		self.idx_train = idx_train
		self.idx_val = idx_val
		self.idx_test = idx_test
		self.mask_train_all = mask_train_all
		self.mask_train = mask_train
		self.mask_val = mask_val
		self.mask_test = mask_test
		#self.num_samp = x.size(0)
		#self.num_feature = x.size(1)
		#self.num_class = int(torch.max(y)) + 1

	def to(self, device):
		self.x = self.x.to(device)
		self.y = self.y.to(device)
		self.feature_type = self.feature_type
		self.feature_idx = self.feature_idx
		self.mask_train_all = self.mask_train_all.to(device)
		self.mask_train = self.mask_train.to(device)
		self.mask_val = self.mask_val.to(device)
		self.mask_test = self.mask_test.to(device)
		return self
