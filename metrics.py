import torch
import random
import numpy as np
import scipy.sparse as sp
import os
import sys
import networkx as nx
import sklearn
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import ShuffleSplit


## All metrics for model eval (benchmark)
def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
			enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),
			dtype=np.int32)
	return labels_onehot


def benchmark(output, idx, true_labels):
	predicted_labels = np.argmax(output[idx].detach().cpu().numpy(), axis = 1)
	true_labels_onehot = encode_onehot(true_labels)
	tp = np.sum((true_labels == 1) & (predicted_labels == 1))
	fp = np.sum((true_labels == 0) & (predicted_labels == 1))
	fn = np.sum((true_labels == 1) & (predicted_labels == 0))
	tn = np.sum((true_labels == 0) & (predicted_labels == 0))
	print("***** TP, TN, FP, FN = [",tp,",",tn,",",fp,",",fn,"] *****")
	AUC = roc_auc_score(true_labels_onehot, output[idx].detach().cpu().numpy())
	recall = recall_score(true_labels, predicted_labels)
	specificity = tn / (tn + fp)
	accuracy = accuracy_score(true_labels, predicted_labels)
	precision = precision_score(true_labels, predicted_labels)
	sensitivity = recall
	f1 = f1_score(true_labels, predicted_labels)
	#recall_0 = tp / (tp + fn)
	#accuracy_0 = (tp + tn) / (tp + tn + fp + fn)
	#precision_0 = tp / (tp + fp)
	#f1_0 = 2 * tp / (2 * tp + fp + fn)
	return (f1, AUC, accuracy, precision, recall, specificity)


def accuracy(output, label):
	""" Return accuracy of output compared to label.
	Parameters
	----------
	output:
		output from model (torch.Tensor)
	label:
		node label (torch.Tensor)
	"""
	preds = output.max(1)[1].type_as(label)
	correct = preds.eq(label).double()
	correct = correct.sum()
	return correct / len(label)


