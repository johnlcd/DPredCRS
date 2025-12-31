import sys,os
import time
from copy import deepcopy
import math
import torch
import numpy as np
import pandas as pd
from collections import Counter
import sklearn
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibrationDisplay
import dpctl
from sklearnex import patch_sklearn, config_context, unpatch_sklearn
from utils_rfe import seed_everything
from metrics import benchmark,accuracy
import daal4py as d4p


"""Machine learning Model
Parameters
----------
data:
	input feature and labels
"""
class ML:
	def __init__(self, args):
		self.args = args
		self.dataset = args.dataset
		self.rapt = args.rapt
		self.seed = args.seed
		self.func = args.func
		self.classifier = args.classifier
		self.rfe_classifier = args.rfe_classifier
		self.gs_classifier = args.gs_classifier
		self.scorer = args.scorer
		self.nest = args.nest
		self.mdepth = args.mdepth
		self.min_fs = args.min_fs
		self.step = args.step
		self.nf_select = args.nf_select
		self.device = args.device
		self.data_path = args.dpath
		self.model_path = "/home/chenjiabin/project/DVT/model_pred/code"
		self.result_path = args.rpath
		if args.model_save_file == "None":
			self.model_save_file = None
		else:
			self.model_save_file = args.model_save_file
	
	# Multinomial Naive Bayes Classifier (train)
	def naive_bayes_classifier(self):
		from sklearn.naive_bayes import GaussianNB
		model = GaussianNB()
		return model
	 
	 
	# KNN Classifier (train)
	def knn_classifier(self):
		from sklearn.neighbors import KNeighborsClassifier
		model = KNeighborsClassifier()
		return model
	 
	# Logistic Regression Classifier (RFE, train)
	def logistic_regression_classifier(self):
		from sklearn.linear_model import LogisticRegression
		model = LogisticRegression(penalty='l2',max_iter=10000)
		return model
	 
	# Random Forest Classifier (RFE,train)
	def random_forest_classifier(self):
		from sklearn.ensemble import RandomForestClassifier
		model = RandomForestClassifier(n_estimators = self.nest, max_depth = self.mdepth, class_weight="balanced", random_state=self.seed)
		return model
	 
	# Random Forest Classifier with GridSearch
	def random_forest_classifier_gs(self, x, y, ds, gp, save_path='./gridsearch_results/'):
		import os
		import pandas as pd
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.model_selection import GridSearchCV
		os.makedirs(save_path, exist_ok=True)

		param_grid = {
			'n_estimators': [100, 200, 300, 500],
			'max_depth': [5, 10, 15, 20, 25, 30]#,
		}
		
		model0 = RandomForestClassifier(class_weight="balanced", random_state=self.seed)
		grid_search = GridSearchCV(
			estimator=model0, param_grid=param_grid, cv=5, 
			n_jobs=3, verbose=1, scoring=self.scorer, return_train_score=True #refit=True
		)
		grid_search.fit(x, y)
		best_parameters = grid_search.best_params_

		results = []
		for i, params in enumerate(grid_search.cv_results_['params']):
			result = params.copy()
			result['mean_test_score'] = grid_search.cv_results_['mean_test_score'][i]
			result['std_test_score'] = grid_search.cv_results_['std_test_score'][i]
			result['mean_train_score'] = grid_search.cv_results_['mean_train_score'][i]
			result['std_train_score'] = grid_search.cv_results_['std_train_score'][i]
			result['rank_test_score'] = grid_search.cv_results_['rank_test_score'][i]
			result['Dataset'] = ds
			result['RAPT'] = self.rapt
			result['Group'] = gp
			result['Seed'] = self.seed
			results.append(result)
		results_df = pd.DataFrame(results)
		results_df.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_rf_all_results_S{self.seed}.csv', index=False)
		
		# Top performing param combination
		top_n = 10
		top_results = results_df.nlargest(top_n, 'mean_test_score')
		top_results.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_rf_top_{top_n}_results_S{self.seed}.csv', index=False)
		
		# Save numerical params to file (output)
		numerical_params = ['n_estimators', 'max_depth', 'min_samples_split']
		for param in numerical_params:
			if param in results_df.columns:
				param_summary = results_df.groupby(param).agg({
					'mean_test_score': ['mean', 'std', 'count'],
					'mean_train_score': ['mean', 'std']
				}).reset_index()
				param_summary['Dataset'] = ds
				param_summary['RAPT'] = self.rapt
				param_summary['Group'] = gp
				param_summary['Seed'] = self.seed
				param_summary.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_rf_param_{param}_analysis_S{self.seed}.csv', index=False)
		
		# Best params
		best_results = {
			'best_score': grid_search.best_score_,
			'best_params': grid_search.best_params_,
			'search_range': param_grid,
			'scoring_metric': self.scorer,
			'cv_folds': 5,
			'total_combinations': len(grid_search.cv_results_['params'])
		}
		print("### Best parameters:")
		print(best_results)
		
		# Save best params
		best_df = pd.DataFrame([{
			'parameter': k,
			'final_value': str(v) if v is not None else 'None',
			'search_range': str(param_grid.get(k, 'N/A')),
			'Dataset': ds,
			'RAPT': self.rapt,
			'Group': gp,
			'Seed': self.seed
		} for k, v in grid_search.best_params_.items()])
		best_df.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_rf_final_parameters_S{self.seed}.csv', index=False)
		
		print(f"\nResults exported to: {save_path}")
		print("Files created:")
		print("  1. rf_all_results.csv - All parameter combinations with scores")
		print("  2. rf_top_10_results.csv - Top 10 performing combinations")
		print("  3. rf_param_*.csv - Analysis for each numerical parameter")
		print("  4. rf_final_parameters.csv - Final selected parameters")
		
		# Best model
		return grid_search.best_estimator_
	 
	 
	# Decision Tree Classifier (train)
	def decision_tree_classifier(self):
		from sklearn import tree
		model = tree.DecisionTreeClassifier()
		return model
	 
	# GBDT(Gradient Boosting Decision Tree) Classifier (train)
	def gradient_boosting_classifier(self):
		from sklearn.ensemble import GradientBoostingClassifier
		model = GradientBoostingClassifier(n_estimators=200)
		return model
	 
	 
	# SVM Classifier with GridSearch (RFE)
	def svm_classifier_gs(self, k, x, y, ds, gp, save_path='./gridsearch_results/'):
		import os
		import pandas as pd
		from sklearn.svm import SVC
		from sklearn.model_selection import GridSearchCV
		from sklearn.preprocessing import StandardScaler
		
		os.makedirs(save_path, exist_ok=True)
		
		# Scale features
		scaler = StandardScaler()
		x_scaled = scaler.fit_transform(x)
		
		# k for kernel
		if k == 'rbf':
			param_grid = [
				{
					'kernel': ['rbf'],
					'C': [0.01, 0.1, 1, 10, 100],
					'gamma': [0.001, 0.01, 0.1, 'scale'],
					'class_weight': ['balanced']
				}
			]
			total_combinations = 5 * 5  # 25
		elif k == 'linear':
			param_grid = [
				{
					'kernel': ['linear'],
					'C': [0.01, 0.1, 1, 10, 100],
					'class_weight': ['balanced']
				}
			]
			total_combinations = 5  # 5
		elif k == 'poly':
			param_grid = [
				{
					'kernel': ['poly'],
					'C': [0.1, 1, 10, 100],
					'gamma': [0.01, 0.1, 'scale'],
					'degree': [2, 3, 4],
					'class_weight': ['balanced']
				}
			]
			total_combinations = 4 * 3 * 3  # 36
		#elif k == 'sigmoid':
			#param_grid = [
				#{
				#	'kernel': ['sigmoid'],
				#	'C': [0.1, 1, 10, 100],
				#	'gamma': [0.01, 0.1, 'scale'],
				#	'class_weight': ['balanced']
				#}
			#]
		elif k == 'all':
			param_grid = [
				# RBF
				{
					'kernel': ['rbf'],
					'C': [0.01, 0.1, 1, 10, 100],
					'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
					'class_weight': ['balanced']
				},
				# linear
				{
					'kernel': ['linear'],
					'C': [0.01, 0.1, 1, 10, 100],
					'class_weight': ['balanced']
				},
				# poly
				{
					'kernel': ['poly'],
					'C': [0.1, 1, 10, 100],
					'gamma': [0.01, 0.1, 'scale'],
					'degree': [2, 3, 4],
					'class_weight': ['balanced']
				}
			]
			total_combinations = (5 * 5) + 5 + (4 * 3 * 3)  # 66

		else:
			raise ValueError(f"Unsupported kernel type: {k}. Use 'linear', 'rbf', 'poly', 'sigmoid' or 'all'.")
		
		print(f">>> Start {k.upper()} kernel GridSearch, total combinations: {total_combinations}")
		
		model0 = SVC(random_state=self.seed, probability=True)
		grid_search = GridSearchCV(
			estimator=model0, 
			param_grid=param_grid,
			cv=5, 
			n_jobs=3, 
			verbose=1, 
			scoring=self.scorer,
			return_train_score=True#,
		)
		
		grid_search.fit(x_scaled, y)
		best_parameters = grid_search.best_params_
		
		# Save all results
		results = []
		for i, params in enumerate(grid_search.cv_results_['params']):
			result = params.copy()
			result['mean_test_score'] = grid_search.cv_results_['mean_test_score'][i]
			result['std_test_score'] = grid_search.cv_results_['std_test_score'][i]
			result['mean_train_score'] = grid_search.cv_results_['mean_train_score'][i]
			result['std_train_score'] = grid_search.cv_results_['std_train_score'][i]
			result['rank_test_score'] = grid_search.cv_results_['rank_test_score'][i]
			result['Dataset'] = ds
			result['RAPT'] = self.rapt
			result['Group'] = gp
			result['Seed'] = self.seed
			results.append(result)
		
		results_df = pd.DataFrame(results)
		results_df.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_svm_{k}_all_results_S{self.seed}.csv', index=False)
		
		# Top performing param combination
		top_n = 10
		top_results = results_df.nlargest(top_n, 'mean_test_score')
		top_results.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_svm_{k}_top_{top_n}_results_S{self.seed}.csv', index=False)
		
		# Analysis by kernel types
		if 'kernel' in results_df.columns:
			for kernel_type in results_df['kernel'].unique():
				kernel_results = results_df[results_df['kernel'] == kernel_type]
				if len(kernel_results) > 0:
					if kernel_type == 'linear':
						group_cols = ['C', 'class_weight']
					elif kernel_type == 'poly':
						group_cols = ['C', 'gamma', 'degree', 'class_weight']
					else:  # rbf
						group_cols = ['C', 'gamma', 'class_weight']
					
					kernel_summary = kernel_results.groupby(group_cols).agg({
						'mean_test_score': ['mean', 'std', 'count'],
						'mean_train_score': ['mean', 'std']
					}).reset_index()
					kernel_summary['kernel'] = kernel_type
					kernel_summary['Dataset'] = ds
					kernel_summary['RAPT'] = self.rapt
					kernel_summary['Group'] = gp
					kernel_summary['Seed'] = self.seed
					kernel_summary.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_svm_{k}_kernel_{kernel_type}_analysis_S{self.seed}.csv', index=False)
		
		# Best params
		best_results = {
			'best_score': grid_search.best_score_,
			'best_params': grid_search.best_params_,
			'scoring_metric': self.scorer,
			'cv_folds': 5,
			'total_combinations': len(grid_search.cv_results_['params'])
		}
		
		print(f"\n### Best parameters of {k.upper()} SVM:")
		for key, value in best_results['best_params'].items():
			print(f"  {key}: {value}")
		print(f"### Best score: {best_results['best_score']:.4f}")
		print(f"### Total combinations: {best_results['total_combinations']}")
		
		# Save best params
		best_df = pd.DataFrame([{
			'parameter': k_param,
			'final_value': str(v) if v is not None else 'None',
			'Dataset': ds,
			'RAPT': self.rapt,
			'Group': gp,
			'Seed': self.seed
		} for k_param, v in grid_search.best_params_.items()])
		best_df.to_csv(f'{save_path}{ds}_R{self.rapt}_G{gp}_svm_{k}_final_parameters_S{self.seed}.csv', index=False)
	
		print(f"\nResults exported to: {save_path}")
		print("Files created:")
		print(f"  1. svm_{k}_all_results.csv")
		print(f"  2. svm_{k}_top_10_results.csv")
		print(f"  3. svm_{k}_kernel_*_analysis.csv")
		print(f"  4. svm_{k}_final_parameters.csv")
		
		final_model = grid_search.best_estimator_
		return final_model, scaler

	# Linear SVM Classifier (RFE, train)
	def lsvm_classifier(self):
		from sklearn.svm import LinearSVC
		from sklearn.svm import SVC
		model = SVC(kernel="linear",C=1,class_weight='balanced',probability=True)
		return model
	
	# SVM Classifier with Sigmoid kernel
	def ssvm_classifier(self):
		from sklearn.svm import SVC
		from sklearn.preprocessing import StandardScaler
		model = SVC(kernel='sigmoid',C=1,class_weight='balanced',probability=True)
		return model
	 
	# SVM Classifier with RBF kernel
	def rsvm_classifier(self):
		from sklearn.svm import SVC
		from sklearn.preprocessing import StandardScaler
		model = SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced',probability=True)
		return model
	 
	# SVM Classifier with RBF kernel
	def psvm_classifier(self):
		from sklearn.svm import SVC
		from sklearn.preprocessing import StandardScaler
		model = SVC(kernel='poly',C=1,degree=3,class_weight='balanced',probability=True)
		return model
	 
	## model evaluationi
	### single model
	def model_eval_single(self, model, x, y): # remove is_binary_class
		predict = model.predict(x)
		ns = x.shape[0]
		pred_prob = model.predict_proba(x)[:, 1]
		cm = confusion_matrix(y, predict)
		accuracy = accuracy_score(y, predict)
		precision = precision_score(y, predict)
		recall = recall_score(y, predict)
		F1_score = f1_score(y, predict)
		AUC_score = roc_auc_score(y, pred_prob)
		return predict, pred_prob, cm, [F1_score, AUC_score, accuracy, precision, recall, ns]

	### combined model
	def model_eval_comb(self, model1, model2, x1, y1, x2, y2):
		predict1 = model1.predict(x1)
		samp_count1 = x1.shape[0]
		pred_prob1 = model1.predict_proba(x1)[:,]
		pred_prob1_bi = model1.predict_proba(x1)[:,1]
		predict2 = model2.predict(x2)
		samp_count2 = x2.shape[0]
		pred_prob2 = model2.predict_proba(x2)[:, ]
		pred_prob2_bi = model2.predict_proba(x2)[:,1]
		y_comb = np.hstack((y1, y2))
		samp_count_comb = samp_count1 + samp_count2
		predict_comb = np.hstack((predict1, predict2))
		pred_prob_comb = np.hstack((pred_prob1_bi,pred_prob2_bi))
		cm_comb = confusion_matrix(y_comb, predict_comb)
		accuracy_comb = accuracy_score(y_comb, predict_comb)
		precision_comb = precision_score(y_comb, predict_comb)
		recall_comb = recall_score(y_comb, predict_comb)
		F1_score_comb = f1_score(y_comb, predict_comb)
		AUC_score_comb = roc_auc_score(y_comb, pred_prob_comb)

		## Brier score and plot calibration curve
		# X_test, y_test
		# y_pred_proba: True
		# Calculate Brier score (0-1, the little the better, "0" for optimal)
		brier_score = brier_score_loss(y_comb, pred_prob_comb)
		print(f"Brier Score: {brier_score:.4f}")
		
		# plot calibration curve
		prob_true, prob_pred = calibration_curve(
			y_comb, 
			pred_prob_comb,
			n_bins=10,
			strategy='quantile'
		)

		plt.figure(figsize=(10, 8))
		plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Ideal calibration")
		plt.plot(prob_pred, prob_true, 'o-', linewidth=3, markersize=10, label=f"Our Model")
		plt.xlabel('Mean Predicted Probability', fontsize=16, fontweight='bold')
		plt.ylabel('Observed Positives', fontsize=16, fontweight='bold')
		plt.title(f'Calibration Curve of {self.dataset} Dataset with {self.classifier} Classifier', fontsize=18, fontweight='bold')
		plt.legend(loc='lower right', fontsize=15, frameon=True, framealpha=0.9, 
				prop={'weight': 'bold', 'size': 15})
		plt.tick_params(axis='both', which='major', labelsize=14)
		plt.tick_params(axis='both', which='minor', labelsize=12)
		plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])

		# Text annotation
		textstr_cal = f"Brier Score: {brier_score:.3f}\n"
		if brier_score < 0.1:
			calibration_level = "Excellent calibration"
			color = 'green'
		elif brier_score < 0.2:
			calibration_level = "Good calibration"
			color = 'orange'
		else:
			calibration_level = "Acceptable calibration"
			color = 'red'
		textstr_cal += calibration_level
		
		plt.text(0.05, 0.95, textstr_cal, transform=plt.gca().transAxes,
				fontsize=15, verticalalignment='top', fontweight='bold',
				bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor=color, linewidth=2))
	
		plt.tight_layout()
		plt.savefig(f'results/figure_plot/{self.dataset}_R{self.rapt}_{self.classifier}_S{self.seed}_calibration_curve.pdf', dpi=300, bbox_inches='tight')
		plt.show()

		## Decision Curve Analysis (DCA) analysis
		"""
		Parameters:
		-----------
		Model (Combined)
		X: Feature (not required)
		y: Label (combined)
		thresholds: Probability Thresholds
		
		Returns:
		--------
		net_benefits: Net Benefit
		all_treat_nb: Net Benefit of Treat All Strategy
		none_treat_nb: Net Benefit of Treat None Strategy
		"""
		# predicted probability (combined: pred_prob_comb)
		
		# Calculate percentage of each type
		prevalence = np.mean(y_comb)
		net_benefits = []
		all_treat_nb = []
		none_treat_nb = []
		thresholds = np.arange(0.0, 1.01, 0.01)

		for threshold in thresholds:
			if threshold == 0:
				threshold = 0.0001  # avoid devided by 0
			
			# classfy into 2 classes by threshold
			y_pred = (pred_prob_comb >= threshold).astype(int)
			
			# confusion matrix
			tn, fp, fn, tp = confusion_matrix(y_comb, y_pred).ravel()
			
			# calculate net benefits
			n = len(y_comb)
			net_benefit = (tp/n) - (fp/n) * (threshold/(1-threshold))
			net_benefits.append(net_benefit)
			
			# "Treat All": predict all sample as positive
			# TP = all positive, FP = all negative
			tp_all = np.sum(y_comb)
			fp_all = np.sum(1-y_comb)
			net_benefit_all = (tp_all/n) - (fp_all/n) * (threshold/(1-threshold))
			all_treat_nb.append(net_benefit_all)
			
			# "Treat None": benefits = 0
			none_treat_nb.append(0)

		net_benefits, all_treat_nb, none_treat_nb = np.array(net_benefits), np.array(all_treat_nb), np.array(none_treat_nb)
		valid_mask = ~np.isnan(net_benefits) & ~np.isinf(net_benefits)
		net_benefits = net_benefits[valid_mask]
		thresholds = thresholds[valid_mask]
		all_treat_nb = all_treat_nb[valid_mask]
		none_treat_nb = none_treat_nb[valid_mask]
		print('\n>>> "Net Benefits" :\n')
		print(net_benefits)
		print('\n>>> "Treat All" :\n')
		print(all_treat_nb)
		print('\n>>> "Treat None" :\n')
		print(none_treat_nb)
			
		# plot decision curve
		plt.figure(figsize=(10, 8))
		plt.plot(thresholds, net_benefits, color='blue', linewidth=3, linestyle='-', label=f'Our Model', alpha=0.9, zorder=3)
		# "Treat All Strategy"
		plt.plot(thresholds, all_treat_nb, color='red', linewidth=2, linestyle='--', label='Treat All', alpha=0.7, zorder=2)
		# "Treat None Strategy"
		plt.plot(thresholds, none_treat_nb, color='black', linewidth=2, linestyle='--', label='Treat None', alpha=0.7, zorder=2)

		# positive benefit region
		positive_indices = np.where(net_benefits > 0)[0]
		print("\n>>> Positive indices:\n")
		print(positive_indices)
		if len(positive_indices) > 0:
			start_thresh = thresholds[positive_indices[0]]
			end_thresh = thresholds[positive_indices[-1]]
			print(f"\n>>> Threshold range with positive net benefit: {start_thresh:.2f} - {end_thresh:.2f}\n")
	
			mid_point = (start_thresh + end_thresh) / 2
			max_nb_in_range = np.max(net_benefits[positive_indices])
	
			textstr_mp = f'Positive NB Range:\n{start_thresh:.1%}-{end_thresh:.1%}'
			plt.text(mid_point+0.2, max_nb_in_range/2+0.25, textstr_mp, transform=plt.gca().transAxes, 
					fontsize=15, verticalalignment='top', fontweight='bold',
					bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='red', linewidth=2))
		
	
		# Max Net Benefit point
		max_nb_idx = np.argmax(net_benefits)
		max_nb_thresh = thresholds[max_nb_idx]
		max_nb_value = net_benefits[max_nb_idx]
		print(f'\n>>> Max Net Benefit (Thresholds):\n    [ {max_nb_value:.3f} ({max_nb_thresh:.1%}) ]\n')
	
		# Clinical thresholds points (10%, 20%)
		clinical_thresholds = [0.1, 0.2, 0.3]
		CT ={}
		for ct in clinical_thresholds:
			idx = np.argmin(np.abs(thresholds - ct))
			nb_value = net_benefits[idx]
	
			plt.scatter(ct, nb_value, color='green', s=75, zorder=4)
			plt.text(ct+0.09, nb_value+0.02, f'{nb_value:.3f} ({ct:.1%})', ha='center', fontsize=14,
					bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue', linewidth=2))
			CT[ct] = nb_value
		print(f'\n>>> Net Bebefits of Clinical Thresholds ([10%, 20%, 30%]):\n')
		print(CT.values())
		
		plt.xlabel('Threshold Probability', fontsize=16, fontweight='bold')
		plt.ylabel('Net Benefit', fontsize=16, fontweight='bold')
		plt.title(f'Decision Curve of {self.dataset} Dataset with {self.classifier} Classifier', fontsize=18, fontweight='bold')
		plt.legend(loc='upper right', fontsize=15, frameon=True, framealpha=0.9, 
				prop={'weight': 'bold', 'size': 15})
		plt.tick_params(axis='both', which='major', labelsize=14)
		plt.tick_params(axis='both', which='minor', labelsize=12)
		plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
		plt.xlim([0, 1])
		plt.ylim([-0.1, 0.9])

		# mark clinical threshold range
		plt.tight_layout()
		plt.savefig(f'results/figure_plot/{self.dataset}_R{self.rapt}_{self.classifier}_S{self.seed}_decision_curve.pdf', dpi=300, bbox_inches='tight')
		plt.show()

		return predict_comb, pred_prob_comb, cm_comb, [F1_score_comb, AUC_score_comb, accuracy_comb, precision_comb, recall_comb, samp_count_comb, samp_count1, samp_count2, brier_score]

	## Brier score and plot calibration curve
	def plot_calibration_curve(self, y_test, y_pred_proba, ds, rapt, classifier):
		# X_test, y_test
		# y_pred_proba: True
		
		# Calculate Brier score (0-1, the little the better, "0" for optimal)
		brier_score = brier_score_loss(y_test, y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba)
		print(f"Brier Score: {brier_score:.4f}")
		
		# plot calibration curve
		prob_true, prob_pred = calibration_curve(
			y_test, 
			y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba,
			n_bins=10,
			strategy='quantile'
		)

		plt.figure(figsize=(8, 6))
		plt.plot(prob_pred, prob_true, 's-', label='Our Model', linewidth=2)
		plt.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated')
		plt.xlabel('Mean Predicted Probability', fontsize=12)
		plt.ylabel('Fraction of Positives', fontsize=12)
		plt.title(f'Calibration Curve (Brier Score = {brier_score:.3f})', fontsize=14)
		plt.legend(loc='best')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f'figure_plot/{ds}_R{rapt}_{classifier}_calibration_curve.pdf', dpi=300)
		plt.show()
		
		# Report in detail
		from sklearn.calibration import CalibrationDisplay
		
		fig, ax = plt.subplots(figsize=(8, 6))
		CalibrationDisplay.from_estimator(
			model, X_test, y_test,
			n_bins=10, strategy='quantile',
			ax=ax, name='Our Model'
		)
		plt.title(f'Calibration Plot of {ds} ({classifier})')
		plt.tight_layout()
		plt.savefig(f'{ds}_R{rapt}_{classifier}_calibration_plot_detailed.pdf', dpi=300)
		plt.show()

	## Decision Curve Analysis (DCA)
	def decision_curve_analysis(self, model, X, y, thresholds=np.arange(0.0, 1.01, 0.01)):
		"""
		Decision Curve Analysis
		
		Parameters:
		-----------
		model : trained model
		X : feature of test set
		y : label of test set (0/1)
		thresholds : probability range thresholds
		
		Returns:
		--------
		net_benefits: net benefit
		all_treat_nb : "Treat All Strategy" benefits
		none_treat_nb : "Treat None Strategy" benefits
		"""
		# predict probability
		y_pred_proba = model.predict_proba(X)[:, 1]
		prevalence = np.mean(y)
		
		net_benefits = []
		all_treat_nb = []
		none_treat_nb = []
		
		for threshold in thresholds:
			if threshold == 0:
				threshold = 0.0001
			
			y_pred = (y_pred_proba >= threshold).astype(int)
			tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
			n = len(y)
			net_benefit = (tp/n) - (fp/n) * (threshold/(1-threshold))
			net_benefits.append(net_benefit)
			tp_all = np.sum(y)
			fp_all = np.sum(1-y)
			net_benefit_all = (tp_all/n) - (fp_all/n) * (threshold/(1-threshold))
			all_treat_nb.append(net_benefit_all)
			none_treat_nb.append(0)
		
		return np.array(net_benefits), np.array(all_treat_nb), np.array(none_treat_nb), thresholds

	
	def plot_decision_curve(self, net_benefits, all_treat_nb, none_treat_nb, thresholds, model_name="Our Model"):
		plt.figure(figsize=(10, 6))
		plt.plot(thresholds, net_benefits, 'b-', linewidth=2, label=model_name)
		plt.plot(thresholds, all_treat_nb, 'k--', linewidth=1.5, label='Treat All')
		plt.plot(thresholds, none_treat_nb, 'k:', linewidth=1.5, label='Treat None')
		
		plt.xlabel('Threshold Probability', fontsize=12)
		plt.ylabel('Net Benefit', fontsize=12)
		plt.title('Decision Curve Analysis', fontsize=14)
		plt.legend(loc='upper right')
		plt.grid(True, alpha=0.3)
		plt.xlim([0, 1])
		
		plt.fill_betweenx([min(net_benefits), max(net_benefits)], 
						  0.05, 0.35, alpha=0.1, color='gray', 
						  label='Clinical Range')
		
		plt.tight_layout()
		plt.show()
	
			
	## Patching scikit-learn with Intel Extension for Scikit-learn (sklearnex) (GPU)
	def pch_sklearn(self,model,x,y,device="gpu:0",n_kernal=10):
		print("\n>>>>> Patching Scikit-learn using \"sklearnex\" (Intel® Extension for Scikit-learn)")
		d4p.daalinit(n_kernal)
		patch_sklearn()
		if self.device.type != "cuda":
			device = "auto"
		print("##### Device (target_offload):\n      [ \"{}\" (\"{}\") ]\n".format(self.device,device))
		with config_context(target_offload=device):
			clustering = model.fit(x,y)
		unpatch_sklearn()
		return model

	## write out rfecv score to file
	def score_to_file(self,file,nfeature_idx,score,score_se,dataset,rapt,group,clf):
		nscore = len(score)
		score_dat = {"Nfeature":nfeature_idx,"CV_Score":score,"CV_Score_SE":score_se}
		score_df = pd.DataFrame(score_dat)
		score_df["DataSet"] = dataset
		score_df["RAPT"] = rapt
		score_df["Group"] = group
		score_df["Classifier"] = clf
		score_df.to_csv(file, sep="\t", header=True, index=False)


	## feature selection (Random Forest)
	def rfe_cv(self, data):
		dataset, rapt, group = data.dataset, self.rapt, data.group
		data_x, data_y, train_x_all, train_y_all, test_x, test_y =  np.array(data.x.cpu()), np.array(data.y.cpu()), np.array(data.x[data.idx_train_all,].cpu()), np.array(data.y[data.mask_train_all].cpu()), np.array(data.x[data.idx_test,].cpu()), np.array(data.y[data.mask_test].cpu())
		fea_label, fea_idx = data.feature_type, data.feature_idx
		num_train_all, num_fea_all = train_x_all.shape
		num_test, num_fea_all = test_x.shape
		is_binary_class = (len(np.unique(train_y_all)) == 2)
		print('\n>>> [ 1 ] Get feature label, flag and index by feature type (\"Clinical\", \"Genomic\", \"Metabolic\") \n...\n... ... \n... ... ...')
		fflag_all = ["Clinical", "Metabolic", "Genomic"]
		d_flag = {"C":"Clinical","M":"Metabolic", "G":"Genomic"}
		fea_all = fea_label.keys()
		flab_all = fea_label.values() ## "label" of all features
		fidx_all = fea_idx.keys()
		flab = list(set(flab_all)) ## "label of feature flags": ["C", "M", "G"]
		d_flab = {}
		d_inx = {}
		for lab in flab:
			fea_lab = []
			ind_lab = []
			for fea in fea_all:
				ind = fea_idx.get(fea)
				if fea_label.get(fea) == lab:
					fea_lab.append(fea)
					ind_lab.append(ind)
			d_flab[lab] = fea_lab
			d_inx[lab] = ind_lab
		print("\n### All features ({}):".format(str(len(list(fea_all)))))
		print(list(fea_all)[0:10],'... ... (Top 10)')

		start_time1 = time.time()
		print('\n\n>>> [ 2 ] Feature selection by importance and return feature index \n... \n... ... \n... ... ...')
		rfe_classifiers = {
				'LR':self.logistic_regression_classifier,
				'RF':self.random_forest_classifier_gs,
				'SVM':self.svm_classifier_gs
				}
		svm_kernels = {'LSVM':'linear',
				'RSVM':'rbf',
				'PSVM':'poly'}
		## Random Forest Classifier
		print('\n******************* \"{}\" Classifier (RFE) ********************'.format(self.rfe_classifier))
		if (self.rfe_classifier == "RF"):
			if (self.gs_classifier == "RF"):
				print('\n>>> [ 2.1 ] HyperParameter optimation with GridSearch (RFE/GridSearch): [ \"{}\" / \"{}\" ]\n...'.format('RF', 'RF'))
			elif (self.gs_classifier in ['LSVM','RSVM','PSVM']):
				print('\n>>> [ 2.1 ] HyperParameter optimation with GridSearch (RFE/GridSearch): [ \"{}\" / \"{}\" ]\n...'.format('RF', self.gs_classifier))
				#self.svm_classifier_gs(svm_kernels.get(self.gs_classifier), data_x, data_y, dataset, group, save_path='./gridsearch_rfe/')
				self.svm_classifier_gs(svm_kernels.get(self.gs_classifier), train_x_all, train_y_all, dataset, group, save_path='./gridsearch_rfe/')
			else:
				print('\n>>> [ 2.1 ] HyperParameter optimation with GridSearch (RFE/GridSearch): [ \"{}\" / \"{}\" ] (Required \"{}\" NOT USED)\n...'.format('RF', 'RF', self.gs_classifier))
			clf = self.random_forest_classifier_gs(train_x_all, train_y_all, dataset, group, save_path='./gridsearch_rfe/')
		elif (self.rfe_classifier == "SVM"):
			if (self.gs_classifier in ['RF','LSVM','RSVM','PSVM']):
				print('\n>>> [ 2.1 ] HyperParameter optimation with GridSearch (RFE/GridSearch): [ \"{}\" / \"{}\" ]\n...'.format('SVM', self.gs_classifier))
				if (self.gs_classifier == "RF"):
					self.random_forest_classifier_gs(train_x_all, train_y_all, dataset, group, save_path='./gridsearch_rfe/')
				elif (self.gs_classifier in ['RSVM','PSVM']):
					self.svm_classifier_gs(svm_kernels.get(self.gs_classifier), train_x_all, train_y_all, dataset, group, save_path='./gridsearch_rfe/')
			else:
				print('\n>>> [ 2.1 ] HyperParameter optimation with GridSearch (RFE/GridSearch): [ \"{}\" / \"{}\" ] (Required \"{}\" NOT USED)\n...'.format('SVM', 'LSVM', self.gs_classifier))
			clf, scaler = self.svm_classifier_gs('linear', train_x_all, train_y_all, dataset, group, save_path='./gridsearch_rfe/')
			train_x_all = scaler.transform(train_x_all)
		else:
			clf = rfe_classifiers[self.rfe_classifier]()
		print("\n##### Estimator (Classifier):    ",clf)
		train_x_fs,train_y_fs = train_x_all,train_y_all
		nfea_all = np.array(train_x_all).shape[1]
		fea_all = list(fea_idx.keys())
		fidx_all = list(fea_idx.values())
		STEP = 1 # default step size (initialize = 1)
		if self.step > 0:
			STEP = math.ceil(self.step) if (self.step >= 1) else math.ceil(self.step*nfea_all)
		print("##### Step Size (Input/Real):     [ {} / {}** ]".format(str(self.step),str(STEP)))
		print("\n>>>>> Feature selection by \"ALL\" features (select and output feature list) ")
		print("====> { RFE(CV) by \"Training Set\" } \n===> \n==> \n=> ")
		if self.func == "rfe":
			## RFE
			model_rfe = RFE(clf,step=STEP,n_features_to_select=self.nf_select,verbose=0)
		elif self.func == "rfecv":
			## RFECV
			NUM_CV = 5
			if (self.rapt == "14"):
				NUM_CV = 3
			cv = StratifiedKFold(NUM_CV,shuffle=True,random_state=self.seed)
			NUM_PARAL = NUM_CV
			model_rfe = RFECV(estimator=clf,step=STEP,cv=cv,min_features_to_select=self.min_fs,scoring=self.scorer,verbose=0,n_jobs=NUM_PARAL) ## "roc_auc"
		# Patching scikit-learn with Intel Extension for Scikit-learn (sklearnex)
		self.pch_sklearn(model_rfe,train_x_fs,train_y_fs)
		# estimator parameters
		print("\n>>>>> Estimator Parameters (RFE) <<<<<")
		print(model_rfe.get_params())
		if self.func == "rfe":
			# feature importance
			print("\n>>>>> Feature Importance (Estimator) <<<<<")
			print(model_rfe.feature_importances_)
			f_i = list(zip(fea_all,model_rfe.feature_importances_))
			f_i.sort(key = lambda x : x[1])
			with PdfPages("{}/fea_sel/Feature_imp_{}_R{}_G{}_{}.pdf".format(self.result_path,dataset,rapt,group,self.rfe_classifier)) as pdf1:
				plt.figure()
				plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
				pdf1.savefig()
				plt.close()
		# feature ranking
		fea_ranking = np.array(model_rfe.ranking_)
		df_fea_rank = pd.DataFrame({"Feature":list(np.array(fea_all)),"Ranking":list(fea_ranking)})
		df_fea_rank_sort = df_fea_rank.sort_values("Ranking",ascending = True)
		path = "{}/fea_sel/{}_R{}_G{}_{}_fea_rank.txt".format(self.result_path,dataset,rapt,group,self.rfe_classifier)
		df_fea_rank_sort.to_csv(path, sep='\t', index=False)
		print("\n>>>>> Feature Ranking <<<<<")
		print(list(fea_ranking))
		# CV results && grid score
		if self.func == "rfecv":
			nfea_idx = list(range(nfea_all,self.min_fs,-STEP))
			nfea_idx+=[self.min_fs] if (self.min_fs not in nfea_idx) else []
			nfea_idx.reverse()
			print("\n>>>>> Index of Feature Number for RFECV <<<<<")
			print(nfea_idx)
			cvs_mean = model_rfe.cv_results_.get("mean_test_score")
			cvs_std = model_rfe.cv_results_.get("std_test_score")
			cvs_se = model_rfe.cv_results_.get("std_test_score")/np.sqrt(NUM_CV)
			print("\n>>>>> CV Mean Test Score <<<<<")
			print(cvs_mean)
			score_file = "{}/rfecv_score/{}_R{}_G{}_{}_CV_score.txt".format(self.result_path,dataset,rapt,group,self.rfe_classifier)
			# write out CV score
			self.score_to_file(score_file,nfea_idx,cvs_mean,cvs_se,dataset,rapt,group,self.rfe_classifier)
			print("\n>>>>> Best Results (score of selected features) <<<<<")
			print("      [ {} ]".format(str(np.round(model_rfe.cv_results_.get("mean_test_score").max(),5))))
			# Plot number of features VS. cross-validation scores
			with PdfPages("{}/rfecv_score/{}_R{}_G{}_{}_CV_score.pdf".format(self.result_path,dataset,rapt,group,self.rfe_classifier)) as pdf2:
				n_scores = len(model_rfe.cv_results_["mean_test_score"])
				plt.figure()
				#  Selected feature number
				plt.xlabel("Number of features selected")
				# CV score
				plt.ylabel("Cross validation score (Mean test accuracy)")
				# plot score of each feature
				plt.errorbar(
						nfea_idx,
						model_rfe.cv_results_["mean_test_score"],
						yerr=model_rfe.cv_results_["std_test_score"],
						)
				plt.title("Recursive Feature Elimination \nwith correlated features")
				pdf2.savefig()
				plt.close()
		# optimal feature number
		print("\n>>>>> Optimal number of features <<<<<\n      [ %d ]" % model_rfe.n_features_)
		# select feature by importance (feature index)
		fea_sel = np.array(fea_all)[model_rfe.support_]
		fidx_sel = np.array(fidx_all)[model_rfe.support_]
		with open("{}/fea_sel/{}_R{}_G{}_{}.select.fea.list".format(self.result_path,dataset,rapt,group,self.rfe_classifier), 'w') as fout_fs:
			for fea in list(fea_sel):
				fout_fs.write(fea + "\n")
		train_x_wrapper = model_rfe.transform(train_x_fs)

		## return selected features (and index)
		print("\n>>>>> Selected feature (N={}) and index <<<<<".format(str(model_rfe.n_features_)))
		print("### Feature (Top 10): \n   ",fea_sel[0:10])
		print("### Feature index: \n   ",fidx_sel)	
		print('\n##### Feature selection took %f s!' % (time.time() - start_time1))
		## save model
		if self.model_save_file != None:
			model_save = (model_rfe,fea_sel,fidx_sel)
			open_file = open("{}/model_save/{}_R{}_G{}_{}_RFE.pkl".format(self.result_path,dataset,rapt,group,self.rfe_classifier), 'wb')
			pkl.dump(model_save, open_file)
			open_file.close()
		return(model_rfe,fea_sel,fidx_sel)


	## train model
	def fit(self, data0, data1, data2, model_rfe0, model_rfe1, model_rfe2, fs0, fs1, fs2):
		dataset, rapt = self.dataset, self.rapt
		data0_x, data0_y, train_x0, train_y0, test_x0, test_y0 = np.array(data0.x.cpu()), np.array(data0.y.cpu()), np.array(data0.x[data0.idx_train_all,].cpu()), np.array(data0.y[data0.mask_train_all].cpu()), np.array(data0.x[data0.idx_test,].cpu()), np.array(data0.y[data0.mask_test].cpu())
		data1_x, data1_y, train_x1, train_y1, test_x1, test_y1 = np.array(data1.x.cpu()), np.array(data1.y.cpu()), np.array(data1.x[data1.idx_train_all,].cpu()), np.array(data1.y[data1.mask_train_all].cpu()), np.array(data1.x[data1.idx_test,].cpu()), np.array(data1.y[data1.mask_test].cpu())
		data2_x, data2_y, train_x2, train_y2, test_x2, test_y2 = np.array(data2.x.cpu()), np.array(data2.y.cpu()), np.array(data2.x[data2.idx_train_all,].cpu()), np.array(data2.y[data2.mask_train_all].cpu()), np.array(data2.x[data2.idx_test,].cpu()), np.array(data2.y[data2.mask_test].cpu())
		fea_label, fea_idx = data1.feature_type, data1.feature_idx
		nfs0, nfs1, nfs2 = len(fs0), len(fs1), len(fs2)
		print("\n##### Number of select features (Group0 / Group1 / Group2):     [ {} / {} / {} ]\n".format(str(nfs0),str(nfs1),str(nfs2)))
		#Transform(Reduce) X to the selected features
		print("\n>>> [ 1 ] Transform of traning/test data to selected features (\"RFECV\") \n... \n... ... \n... ... ...")
		train_x0_t, train_x1_t, train_x2_t = model_rfe0.transform(train_x0), model_rfe1.transform(train_x1), model_rfe2.transform(train_x2)
		test_x0_t, test_x1_t, test_x2_t = model_rfe0.transform(test_x0), model_rfe1.transform(test_x1), model_rfe2.transform(test_x2)
		ntrain0, ntest0 = train_x0_t.shape[0], test_x0_t.shape[0]
		ntrain1, ntest1 = train_x1_t.shape[0], test_x1_t.shape[0]
		ntrain2, ntest2 = train_x2_t.shape[0], test_x2_t.shape[0]
		nfea_t0, nfea_t1, nfea_t2 = train_x0_t.shape[1], train_x1_t.shape[1], train_x2_t.shape[1]
		nfea_all = train_x0.shape[1]
		print("##### Shape of transform data:\n      G0 [ Train , Test ]: [ ({},{}) , ({},{}) ];\n      G1 [ Train , Test ]: [ ({},{}) , ({},{}) ];\n      G2 [ Train , Test ]: [ ({},{}) , ({},{}) ]".format(str(ntrain0),str(nfea_t0),str(ntest0),str(nfea_t0),str(ntrain1),str(nfea_t1),str(ntest1),str(nfea_t1),str(ntrain2),str(nfea_t2),str(ntest2),str(nfea_t2)))
		print(train_x0_t,train_x1_t,train_x2_t)
		print('\n******************** Data Info *********************')
		print('##### Number of each dataset (G0/G1/G2):\n      [ Training data: {}/{}/{} ],\n      [ Testing data: {}/{}/{} ],\n      [ Feature number selected (ALL): {}/{}/{} ({}) ]\n'.format(str(ntrain0),str(ntrain1),str(ntrain2),str(ntest0),str(ntest1),str(ntest2),str(nfs0),str(nfs1),str(nfs2),str(nfea_all)))
		classifiers = {'NB':self.naive_bayes_classifier, 
				'KNN':self.knn_classifier,
				'LR':self.logistic_regression_classifier,
				'RF':self.random_forest_classifier_gs,
				'DT':self.decision_tree_classifier,
				'LSVM':self.svm_classifier_gs,
				'RSVM':self.svm_classifier_gs,
				'PSVM':self.svm_classifier_gs,
				'GBDT':self.gradient_boosting_classifier,
				}
		svm_kernels = {'LSVM':'linear',
				'RSVM':'rbf',
				'PSVM':'poly'}
		print('\n>>> [ 2 ] Training and evaluation modeles (by Group) using Selected Features by \"{}\" \n... \n... ... \n... ... ...'.format(self.func)) #RFECV
		print('\n******************* \"{}\" Classifier ( Training && Evaluation ) ********************'.format(self.classifier))
		start_time2 = time.time()
		if (self.classifier == 'RF'):
			print('\n>>> [ 2.1 ] HyperParameter optimation with GridSearch for classifier \"{}\" \n...'.format(self.classifier))
			model_t0 = self.random_forest_classifier_gs(train_x0_t, train_y0, dataset, 0, save_path='./gridsearch_train/')
			model_t1 = self.random_forest_classifier_gs(train_x1_t, train_y1, dataset, 1, save_path='./gridsearch_train/')
			model_t2 = self.random_forest_classifier_gs(train_x2_t, train_y2, dataset, 2, save_path='./gridsearch_train/')
		elif (self.classifier in ['LSVM','RSVM','PSVM']):
			print('\n>>> [ 2.1 ] HyperParameter optimation with GridSearch for classifier \"{}\" \n...'.format(self.classifier))
			model_t0, scaler0 = self.svm_classifier_gs(svm_kernels.get(self.classifier), train_x0_t, train_y0, dataset, 0, save_path='./gridsearch_train/')
			model_t1, scaler1 = self.svm_classifier_gs(svm_kernels.get(self.classifier), train_x1_t, train_y1, dataset, 1, save_path='./gridsearch_train/')
			model_t2, scaler2 = self.svm_classifier_gs(svm_kernels.get(self.classifier), train_x2_t, train_y2, dataset, 2, save_path='./gridsearch_train/')
			train_x0_t = scaler0.transform(train_x0_t)
			train_x1_t = scaler1.transform(train_x1_t)
			train_x2_t = scaler2.transform(train_x2_t)
			test_x0_t = scaler0.transform(test_x0_t)
			test_x1_t = scaler1.transform(test_x1_t)
			test_x2_t = scaler2.transform(test_x2_t)
		else:
			model_t0 = classifiers[self.classifier]()
			model_t1 = classifiers[self.classifier]()
			model_t2 = classifiers[self.classifier]()
		print('\n===============================  Group 0  ================================')
		model_t0.fit(train_x0_t,train_y0) if self.classifier == "LR" else self.pch_sklearn(model_t0,train_x0_t,train_y0)
		pred_train_0,prob_train_0,cm_train_0,metrics_train_0 = self.model_eval_single(model_t0,train_x0_t,train_y0)
		pred_test_0,prob_test_0,cm_test_0,metrics_test_0 = self.model_eval_single(model_t0,test_x0_t,test_y0)
		print(">>>>> Estimator && Parameters <<<<<")
		print("### Estimator: ",model_t0)
		print('### Parameters: ',model_t0.get_params())
		print('\n>>>>> Model evaluation <<<<<')
		print('--   Dataset    --||-- F1 score  --||----  AUC  ----||----  Acc  ----||----  Pre  ----||----  Rec  ----||--  N Sample  --')
		print('--   Training   --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   {5:04d}   ---'.format(*metrics_train_0), flush = True)
		print('--   Testing    --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   {5:04d}   ---'.format(*metrics_test_0), flush = True)
		print('\n===============================  Group 1  ================================')
		model_t1.fit(train_x1_t,train_y1) if self.classifier == "LR" else self.pch_sklearn(model_t1,train_x1_t,train_y1)
		pred_train_1,prob_train_1,cm_train_1,metrics_train_1 = self.model_eval_single(model_t1,train_x1_t,train_y1)
		pred_test_1,prob_test_1,cm_test_1,metrics_test_1 = self.model_eval_single(model_t1,test_x1_t,test_y1)
		print(">>>>> Estimator && Parameters <<<<<")
		print("### Estimator: ",model_t1)
		print('### Parameters: ',model_t1.get_params())
		print('\n>>>>> Model evaluation <<<<<')
		print('--   Dataset    --||-- F1 score  --||----  AUC  ----||----  Acc  ----||----  Pre  ----||----  Rec  ----||--  N Sample  --')
		print('--   Training   --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   {5:04d}   ---'.format(*metrics_train_1), flush = True)
		print('--   Testing    --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   {5:04d}   ---'.format(*metrics_test_1), flush = True)
		print('\n==========================================================================')
		print('\n===============================  Group 2  ================================')
		model_t2.fit(train_x2_t,train_y2) if self.classifier == "LR" else self.pch_sklearn(model_t2,train_x2_t,train_y2)
		pred_train_2,prob_train_2,cm_train_2,metrics_train_2 = self.model_eval_single(model_t2,train_x2_t,train_y2)
		pred_test_2,prob_test_2,cm_test_2,metrics_test_2 = self.model_eval_single(model_t2,test_x2_t,test_y2)
		print(">>>>> Estimator && Parameters <<<<<")
		print("### Estimator: ",model_t2)
		print('### Parameters: ',model_t2.get_params())
		print('\n>>>>> Model evaluation <<<<<')
		print('--   Dataset    --||-- F1 score  --||----  AUC  ----||----  Acc  ----||----  Pre  ----||----  Rec  ----||--  N Sample  --')
		print('--   Training   --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   {5:04d}   ---'.format(*metrics_train_2), flush = True)
		print('--   Testing    --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   {5:04d}   ---'.format(*metrics_test_2), flush = True)
		print('\n==========================================================================')
		print('\n##### Training took %f s! \n' % (time.time() - start_time2))
		print('\n>>> [ 3 ] Evaluation of \"Combined\" model \n... \n... ... \n... ... ...')
#		print('\n*******************************  Training Set  ******************************** \n... \n... ... \n... ... ...')
		pred_train_comb,prob_train_comb,cm_train_comb,metrics_train_comb = self.model_eval_comb(model_t1,model_t2,train_x1_t,train_y1,train_x2_t,train_y2)
#		print('\n*******************************************************************************')
#		print('\n*******************************  Testing Set  ********************************* \n... \n... ... \n... ... ...')
		pred_test_comb,prob_test_comb,cm_test_comb,metrics_test_comb = self.model_eval_comb(model_t1,model_t2,test_x1_t,test_y1,test_x2_t,test_y2)
#		print('\n******************************************************************************* \n')
		print('\n>>>>> Model evaluation (combined) <<<<<')
		print('--   Dataset    --||-- F1 score  --||----  AUC  ----||----  Acc  ----||----  Pre  ----||----  Rec  ----||- Brier Score -||-- N Sample (ALL/Group1/Group2) --')
		print('--   Training   --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---  {8:.3f}  ---||---   ( {5:03d} / {6:03d} / {7:03d} )   ---'.format(*metrics_train_comb), flush = True)
		print('--   Testing    --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---  {8:.3f}  ---||---   ( {5:03d} / {6:03d} / {7:03d} )   ---'.format(*metrics_test_comb), flush = True)

		# Save model perfomance
		metrics_train_0.append(0)
		metrics_train_1.append(0)
		metrics_train_2.append(0)
		metrics_test_0.append(0)
		metrics_test_1.append(0)
		metrics_test_2.append(0)
		metrics_train_comb_final = metrics_train_comb[0:6] + metrics_train_comb[8:9]
		metrics_test_comb_final = metrics_test_comb[0:6] + metrics_test_comb[8:9]
		perf_train_dat = {"All":list(np.round(metrics_train_0,5)),
				"G1":list(np.round(metrics_train_1,5)),
				"G2":list(np.round(metrics_train_2,5)),
				"Combined":list(np.round(metrics_train_comb_final,5))
				}
		perf_test_dat = {"All":list(np.round(metrics_test_0,5)),
				"G1":list(np.round(metrics_test_1,5)),
				"G2":list(np.round(metrics_test_2,5)),
				"Combined":list(np.round(metrics_test_comb_final,5))
				}
		perf_train_df = pd.DataFrame(perf_train_dat)
		perf_train_df = perf_train_df.T
		perf_train_df.columns = ["F1","AUC","Acc","Pre","Rec","Nsample","Brier"]
		perf_train_df["DataSet"] = dataset
		perf_train_df["Group"] = perf_train_df.index
		perf_train_df["RAPT"] = rapt
		perf_train_df["Classifier"] = self.classifier
		perf_train_df["Model_set"] = "Training"
		perf_train_df["Nfeature"] = [nfea_t0,nfea_t1,nfea_t2,str(nfea_all)+"*"]
		perf_train_df["Seed"] = self.seed
		perf_test_df = pd.DataFrame(perf_test_dat)
		perf_test_df = perf_test_df.T
		perf_test_df.columns = ["F1","AUC","Acc","Pre","Rec","Nsample","Brier"]
		perf_test_df["DataSet"] = dataset
		perf_test_df["Group"] = perf_test_df.index
		perf_test_df["RAPT"] = rapt
		perf_test_df["Classifier"] = self.classifier
		perf_test_df["Model_set"] = "Testing"
		perf_test_df["Nfeature"] = [nfea_t0,nfea_t1,nfea_t2,str(nfea_all)+"*"]
		perf_test_df["Seed"] = self.seed
		perf_df_comb = pd.concat([perf_train_df,perf_test_df]).reset_index(drop=True)
		perf_df_comb["Nsample"] = perf_df_comb["Nsample"].astype("int")
		print("\n>>>>> Summary of all model performance <<<<<")
		print(perf_df_comb)
		perf_file = "{}/model_perf/{}_R{}__{}_{}__S{}.perf.txt".format(self.result_path,dataset,rapt,self.rfe_classifier,self.classifier,self.seed)
		perf_df_comb.to_csv(perf_file, sep="\t", header=True, index=False)
		
		# Save RFE and final models (tuple)
		tuple_model = (model_rfe0,model_rfe1,model_rfe2,model_t0,model_t1,model_t2)
		if self.model_save_file != None:
			open_file = open("{}/model_save/{}_R{}__{}_{}__tuple_model_S{}.pkl".format(self.result_path,dataset,rapt,self.rfe_classifier,self.classifier,self.seed), 'wb')
			pkl.dump(tuple_model, open_file)
			open_file.close()

