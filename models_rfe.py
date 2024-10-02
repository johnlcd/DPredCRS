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
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
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
		self.scorer = args.scorer
		self.nest = args.nest
		self.mdepth = args.mdepth
		self.min_fs = args.min_fs
		self.step = args.step
		self.nf_select = args.nf_select
		self.device = args.device
		self.data_path = args.dpath
		self.result_path = args.rpath
		if args.model_save_file == "None":
			self.model_save_file = None
		else:
			self.model_save_file = args.model_save_file
	
	# Multinomial Naive Bayes Classifier (train)
	def naive_bayes_classifier(self): #train_x, train_y):
		#from sklearn.naive_bayes import MultinomialNB
		#model = MultinomialNB(alpha=0.01)
		from sklearn.naive_bayes import GaussianNB
		model = GaussianNB()
		#model.fit(train_x, train_y)
		return model
	 
	 
	# KNN Classifier (train)
	def knn_classifier(self): #train_x, train_y):
		from sklearn.neighbors import KNeighborsClassifier
		model = KNeighborsClassifier()
		#model.fit(train_x, train_y)
		return model
	 
	# Logistic Regression Classifier (RFE, train)
	def logistic_regression_classifier(self): #train_x, train_y):
		from sklearn.linear_model import LogisticRegression
		model = LogisticRegression(penalty='l2',max_iter=10000)
		#model.fit(train_x, train_y)
		return model
	 
	# Random Forest Classifier (RFE,train)
	def random_forest_classifier(self): #train_x, train_y):
		from sklearn.ensemble import RandomForestClassifier
		model = RandomForestClassifier(n_estimators = self.nest, max_depth = self.mdepth, class_weight="balanced", random_state=self.seed)
		#model = RandomForestClassifier(n_estimators=8)
		#model.fit(train_x, train_y)
		return model
	 
	 
	# Decision Tree Classifier (train)
	def decision_tree_classifier(self): #train_x, train_y):
		from sklearn import tree
		model = tree.DecisionTreeClassifier()
		#model.fit(train_x, train_y)
		return model
	 
	# GBDT(Gradient Boosting Decision Tree) Classifier (train)
	def gradient_boosting_classifier(self): #train_x, train_y):
		from sklearn.ensemble import GradientBoostingClassifier
		model = GradientBoostingClassifier(n_estimators=200)
		#model.fit(train_x, train_y)
		return model
	 
	 
	# Linear SVM Classifier (RFE, train)
	def lsvm_classifier(self): #train_x, train_y):
		from sklearn.svm import LinearSVC
		from sklearn.svm import SVC
		model = SVC(kernel="linear",C=1,probability=True)
		#model = LinearSVC(penalty='l2')
		#model.fit(train_x, train_y)
		return model
	
	# SVM Classifier
	## "RBF" kernel
	def svm_rbf_classifier(self): #train_x, train_y):
		from sklearn.svm import SVC
		model = SVC(kernel='rbf',probability=True)
		#model.fit(train_x, train_y)
		return model
	## "Poly" kernel
	def svm_poly_classifier(self): #train_x, train_y):
		from sklearn.svm import SVC
		model = SVC(kernel='poly',probability=True)
		#model.fit(train_x, train_y)
		return model
	 
	# SVM Classifier using cross validation
	def svm_cross_validation(self): #train_x, train_y):
		from sklearn.grid_search import GridSearchCV
		from sklearn.svm import SVC
		model = SVC(kernel='rbf',probability=True)
		param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
		grid_search = GridSearchCV(model, param_grid, n_jobs = 3, verbose=1)
		grid_search.fit(train_x, train_y)
		best_parameters = grid_search.best_estimator_.get_params()
		for para, val in best_parameters.items():
			print(para, val)
		model = SVC(kernel='rbf',C=best_parameters['C'],gamma=best_parameters['gamma'],probability=True)
		#model.fit(train_x, train_y)
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
#		print("\n##### Shape of y1, y2:")
#		print(y1.shape,y2.shape)
		predict1 = model1.predict(x1)
		samp_count1 = x1.shape[0]
		#pred_prob1 = model1.predict_proba(x1)[:, 1]
		pred_prob1 = model1.predict_proba(x1)[:,]
		pred_prob1_bi = model1.predict_proba(x1)[:,1]
		predict2 = model2.predict(x2)
		samp_count2 = x2.shape[0]
		#pred_prob2 = model2.predict_proba(x2)[:, 1]
		pred_prob2 = model2.predict_proba(x2)[:, ]
		pred_prob2_bi = model2.predict_proba(x2)[:,1]
#		print("\n##### Shape of predict_y1, predict_y2:")
#		print(predict1.shape,predict2.shape)
		#x_comb = np.concatenate((x1, x2), axis=1)
		y_comb = np.hstack((y1, y2))
		samp_count_comb = samp_count1 + samp_count2
		predict_comb = np.hstack((predict1, predict2))
#		print("\n##### Shape of y_comb, predict_comb:")
#		print(y_comb.shape,predict_comb.shape)
#		print("\n##### Shape of predict probability of Group1/2:")
#		print(pred_prob1.shape,pred_prob2.shape)
		pred_prob_comb = np.hstack((pred_prob1_bi,pred_prob2_bi))
		cm_comb = confusion_matrix(y_comb, predict_comb)
		accuracy_comb = accuracy_score(y_comb, predict_comb)
		precision_comb = precision_score(y_comb, predict_comb)
		recall_comb = recall_score(y_comb, predict_comb)
		F1_score_comb = f1_score(y_comb, predict_comb)
		AUC_score_comb = roc_auc_score(y_comb, pred_prob_comb)
		return predict_comb, pred_prob_comb, cm_comb, [F1_score_comb, AUC_score_comb, accuracy_comb, precision_comb, recall_comb, samp_count_comb, samp_count1, samp_count2]

	## Patching scikit-learn with Intel Extension for Scikit-learn (sklearnex) (GPU)
	def pch_sklearn(self,model,x,y,device="gpu:0",n_kernal=10):
		print("\n>>>>> Patching Scikit-learn using \"sklearnex\" (Intel® Extension for Scikit-learn)")
		d4p.daalinit(n_kernal)
		#print(self.device.type)
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
		fflag_all = ["Clinical", "Metabolic", "Genomic"] ## "feature flags": ["Clinical", "Metabolic", "Genomic"]
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
#		print("\n### All feature labels of the dataset:")
#		print(list(flab_all))
		print("\n### All features ({}):".format(str(len(list(fea_all)))))
		print(list(fea_all)[0:10],'... ... (Top 10)')
#		print("\n### Features indexes:")
#		print(list(ind_lab))

		start_time1 = time.time()
		print('\n\n>>> [ 2 ] Feature selection by importance and return feature index \n... \n... ... \n... ... ...')
		rfe_classifiers = {
				'LR':self.logistic_regression_classifier,
				'RF':self.random_forest_classifier,
				'LSVM':self.lsvm_classifier
				}
		## Random Forest Classifier
		#n_est = 150 #number of estimators (trees) default = 100
		#m_dpt = 50 #maximun of tree depth, defult = None
		print('\n******************* \"{}\" Classifier (RFE) ********************'.format(self.rfe_classifier))
		clf = rfe_classifiers[self.rfe_classifier]()
		print("\n##### Estimator (Classifier):    ",clf)
		#train_x_fs,train_y_fs = train_x_all,train_y_all
		#test_x_fs,test_y_fs = test_x,test_y
		#fea_all = np.array(list(fea_idx.keys()))
		#fidx_all = np.array(list(fea_idx.values()))
		train_x_fs,train_y_fs = data_x,data_y
		nfea_all = np.array(data_x).shape[1]
		fea_all = list(fea_idx.keys())
		fidx_all = list(fea_idx.values())
		STEP = 1 # default step size (initialize = 1)
		if self.step > 0:
			STEP = math.ceil(self.step) if (self.step >= 1) else math.ceil(self.step*nfea_all) #math.floor
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
			f_i = list(zip(fea_all,model_rfe.feature_importances_)) ## e.g. features = load_boston()['feature_names']
			f_i.sort(key = lambda x : x[1])
			with PdfPages("{}/fs_res/Feature_imp_{}_R{}_G{}_{}_S{}.pdf".format(self.result_path,dataset,rapt,group,self.rfe_classifier)) as pdf1:
				plt.figure()
				plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
				pdf1.savefig()
				plt.close()
		# feature ranking
		fea_ranking = np.array(model_rfe.ranking_)
		df_fea_rank = pd.DataFrame({"Feature":list(np.array(fea_all)),"Ranking":list(fea_ranking)})
		df_fea_rank_sort = df_fea_rank.sort_values("Ranking",ascending = True)
		path = "{}/fs_res/fea_rank/{}_R{}_G{}_{}_fea_rank_S{}.txt".format(self.result_path,dataset,rapt,group,self.rfe_classifier,self.seed)
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
			score_file = "{}/rfecv_score/{}_R{}_G{}_{}_S{}_CV_score.txt".format(self.result_path,dataset,rapt,group,self.rfe_classifier,self.seed)
			# write out CV score
			self.score_to_file(score_file,nfea_idx,cvs_mean,cvs_se,dataset,rapt,group,self.rfe_classifier)
			print("\n>>>>> Best Results (score of selected features) <<<<<")
			print("      [ {} ]".format(str(np.round(model_rfe.cv_results_.get("mean_test_score").max(),5))))
#			print("\n>>>>> Grid Scores <<<<<")
#			print(model_rfe.grid_scores_)
			# Plot number of features VS. cross-validation scores
			with PdfPages("{}/rfecv_score/{}_R{}_G{}_{}_S{}_CV_score.pdf".format(self.result_path,dataset,rapt,group,self.rfe_classifier,self.seed)) as pdf2:
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
		# write selected features to file ("***_Metabolic_RFE.select.fea.list")
		with open("{}/fs_res/fea_sel/{}_R{}_G{}_{}_S{}.select.fea.list".format(self.result_path,dataset,rapt,group,self.rfe_classifier,self.seed), 'w') as fout_fs:
			for fea in list(fea_sel):
				fout_fs.write(fea + "\n")
		train_x_wrapper = model_rfe.transform(train_x_fs)

		## return selected features (and index)
		print("\n>>>>> Selected feature (N={}) and index <<<<<".format(str(model_rfe.n_features_)))
		print("### Feature (Top 10): \n   ",fea_sel[0:10])
		print("### Feature index: \n   ",fidx_sel)	
		print('\n##### Feature selection took %f s!' % (time.time() - start_time1))
		
		## save model
		fs_model_save = "{}/model_save/{}_R{}_G{}_{}_RFE.pkl".format(self.result_path,dataset,rapt,group,self.rfe_classifier)
		fs_model_bak = "{}/model_save/fs/{}_R{}_G{}_{}_RFE_S{}.pkl".format(self.result_path,dataset,rapt,group,self.rfe_classifier,self.seed)
		if self.model_save_file != None:
			model_save = (model_rfe,fea_sel,fidx_sel)
			#model_save = (data,model_rfe,fea_sel,fidx_sel)
			open_file = open(fs_model_save, 'wb') ##"DVT_UKB_S0_k11_h16"
			pkl.dump(model_save, open_file)
			open_file.close()
			os.system("cp -a " + fs_model_save + " " + fs_model_bak)
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
				'RF':self.random_forest_classifier,
				'DT':self.decision_tree_classifier,
				'LSVM':self.lsvm_classifier,
				'SVMR':self.svm_rbf_classifier,
				'SVMP':self.svm_poly_classifier,
				'SVMCV':self.svm_cross_validation,
				'GBDT':self.gradient_boosting_classifier,
				}
		print('\n>>> [ 2 ] Training and evaluation modeles (by Group) using Selected Features by \"{}\" \n... \n... ... \n... ... ...'.format(self.func)) #RFECV
		print('\n******************* \"{}\" Classifier ( Training && Evaluation ) ********************'.format(self.classifier))
		start_time2 = time.time()
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
		print('--   Dataset    --||-- F1 score  --||----  AUC  ----||----  Acc  ----||----  Pre  ----||----  Rec  ----||-- N Sample (ALL/Group1/Group2) --')
		print('--   Training   --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   ( {5:03d} / {6:03d} / {7:03d} )   ---'.format(*metrics_train_comb), flush = True)
		print('--   Testing    --||---  {0:.3f}  ---||---  {1:.3f}  ---||---  {2:.3f}  ---||---  {3:.3f}  ---||---  {4:.3f}  ---||---   ( {5:03d} / {6:03d} / {7:03d} )   ---'.format(*metrics_test_comb), flush = True)

		# Save model perfomance
		perf_train_dat = {"All":list(np.round(metrics_train_0,5)),
				"G1":list(np.round(metrics_train_1,5)),
				"G2":list(np.round(metrics_train_2,5)),
				"Combined":list(np.round(metrics_train_comb[0:6],5))
				}
		perf_test_dat = {"All":list(np.round(metrics_test_0,5)),
				"G1":list(np.round(metrics_test_1,5)),
				"G2":list(np.round(metrics_test_2,5)),
				"Combined":list(np.round(metrics_test_comb[0:6],5))
				}
		perf_train_df = pd.DataFrame(perf_train_dat)
		perf_train_df = perf_train_df.T
		perf_train_df.columns = ["F1","AUC","Acc","Pre","Rec","Nsample"]
		perf_train_df["DataSet"] = dataset
		perf_train_df["Group"] = perf_train_df.index
		perf_train_df["RAPT"] = rapt
		perf_train_df["Classifier"] = self.classifier
		perf_train_df["Model_set"] = "Training"
		perf_train_df["Nfeature"] = [nfea_t0,nfea_t1,nfea_t2,str(nfea_all)+"*"]
		perf_train_df["Seed"] = self.seed
		perf_test_df = pd.DataFrame(perf_test_dat)
		perf_test_df = perf_test_df.T
		perf_test_df.columns = ["F1","AUC","Acc","Pre","Rec","Nsample"]
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
		#perf_file = "{}/model_perf/{}_R{}__{}_{}__S{}.perf.txt".format(self.result_path,dataset,rapt,self.rfe_classifier,self.classifier,self.seed)
		perf_file = "{}/model_perf/{}_R{}_{}_S{}.perf.txt".format(self.result_path,dataset,rapt,self.classifier,self.seed)
		perf_df_comb.to_csv(perf_file, sep="\t", header=True, index=False)
		
		# Save final models
		final_model = (model_t0,model_t1,model_t2)
		if self.model_save_file != None:
			#open_file = open("{}/model_save/{}_R{}__{}_{}__final_model_S{}.pkl".format(self.result_path,dataset,rapt,self.rfe_classifier,self.classifier,self.seed), 'wb')
			open_file = open("{}/model_save/{}_R{}_{}_final_model_S{}.pkl".format(self.result_path,dataset,rapt,self.classifier,self.seed), 'wb')
			pkl.dump(final_model, open_file)
			open_file.close()

