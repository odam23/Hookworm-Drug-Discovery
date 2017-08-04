#Author : Odame Agyapong aodame004@st.ug.edu.gh
#Supervisor : Dr. Samuel Kwofie (Department of Biomedical Engineering, Engineering -University of Ghana)
#Co-Supervisor : Prof. Michael Wilson (Noguchi Memorial Institute Medical Research)
#All rights reserved 2017
#PCM Model trained on Beta tubulin bioactivity dataset from BindingDB(Mined 20/12/16)
#Molecular Descriptors : 1024bit Morgan Binary Fingerprints (Rdkit) - ECFP4 and Composition/Transition/Distribution Descriptors
#Dependencies : python, sklearn, numpy, pandas

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import uniform as sp_rand
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix,classification_report,matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold,KFold, cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split

global scaler
def BuildDataset():
	global scaler
	dataset=pd.read_csv("dataset.csv")

	array = dataset.values
	X = array[:,0:1005]
	y = array[:,1005]
	
	data_df = pd.DataFrame.from_csv("dataset.csv")
	#data_df = data_df.reindex(np.random.permutation(data_df.index))
	X_values= data_df[:-1]
	print(X_values)
	#print(data_df)

	#scaling the entire data instead of fitting the transformation on the training set
	scaler = preprocessing.StandardScaler().fit(X)
	X = scaler.transform(X)
	#print(dataset.shape)
	test_size = 0.33
	seed = 777

	#Split dataset 
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)
	# standardize the data attributes
	scaler = preprocessing.StandardScaler().fit(X_train)
	
	X_train = scaler.transform(X_train)

	X_train=preprocessing.normalize(X_train)
	X_train=preprocessing.scale(X_train)
        
        return X_train, X_test, y_train, y_test
	
	#########################################PCA


def introMessage():
	print('==============================================================================================')
	print(' Author: Odame Agyapong \n Email: aodame004@st.ug.edu.gh\n Supervisor: Dr. Samuel Kwofie\n Co-supervisor: Prof. Michael Wilson')
	print(' Address: Department of Biomedical Engineering, University of Ghana')
	print(' Title: PCM Model trained on tubulin bioactivity dataset from BindingDB')
	print('==============================================================================================\n')
	return


def Analysis():
	introMessage()
	X_train, X_test, y_train, y_test = BuildDataset()
	###prelim testing with rbf kernel with gamma of 0.1 and C of 1e-01
	clf = SVC(C=1e-01, kernel='rbf', gamma=0.1)
	#clf = SVC(kernel='poly', degree=3, probability=True, max_iter=100000)
	fitter = clf.fit(X_train, y_train)

	y_predictions = clf.predict(X_test)
	#print(metrics.accuracy_score(y_test,y_predictions))

	X_test = scaler.transform(X_test)
	#print("Classifier Score:", clf.score(X_test,y_test))

	#######################################


	#############################################
	#use n_splits instead of n_folds for later version of scikit

	estimator = SVC(C=1e-01, kernel='rbf', gamma=0.1, probability=True)
	#mvs=svm.SVC()
	#parameters = [{'C':[0.1,1,10],'kernel':['rbf','linear']}]
	#estimator = SVC(kernel='poly', degree=3, probability=True, max_iter=100000)
	cv= cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True, random_state=777) 
	#cv= cross_validation.KFold(len(y_train), n_folds=10, shuffle=False, random_state=777) 
	#set up an possible range of valuesfor optimal parameter C. returns 10 evelnly spaced between -10 and -1 i.e 10^-10, 10^-1, param grid is parameters to optimise
	#hyperparameters passed as arguments to the constructor of the estimator classes, typicall C, kernel and gamma

	parameters = {"C":[0.001,0.001,0.1,1,10,100,1000], "gamma":[0.001,0.0001,0.1,1]}
	classify = GridSearchCV(estimator, parameters, cv=cv, scoring='roc_auc', n_jobs=10)
	

	classify.fit(X_train, y_train)

	
	###########Various metric assessment with 
	###Best params
	print (classify.best_score_)
	print (classify.best_params_)
	
	#Predict on Hold-out test set
	predictions = classify.predict(X_test)
	print(predictions)

	#A plot on predictions
	plt.scatter(y_test,predictions)
	plt.xlabel("True Values")
	plt.ylabel("Predictions")
	plt.show()
	
	#Classification Accuracy
	#print("Classifier Score:", classify.score(X_test,y_test))
	#Classification accuracy
	print("Classification Score:", metrics.accuracy_score(y_test,predictions))
        
        #Classification error
        print("Classification Error:", 1-metrics.accuracy_score(y_test,predictions))
 
	#Confusion Matrix
	matrix= confusion_matrix(y_test,predictions)
	print("Our Confusion matrix:",matrix)
        
        TP = matrix[1,1]
        TN = matrix[0,0]
        FP = matrix[0,1]
        FN = matrix[1,0]

        print("sensitivity",metrics.recall_score(y_test,predictions))#TPR or recall
        print("specificity",TN/float(FP + TN))#specificity. How specific or selective

	#Classification report
	report= classification_report(y_test,predictions)
	print(report)

	

	print(1- metrics.accuracy_score(y_test,predictions)) 
	
	#MCC score
	print("Mathews Correlation Coefficient is :", matthews_corrcoef(y_test,predictions))
	##################

	####################################################################

	
	##Plotting the AUC of the classifier
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.title('Receiver Operating Characteristic')
	plt.plot(false_positive_rate, true_positive_rate, 'b',
	label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()
	##################################################################


	######Dumping Model as a pickle
	joblib.dump(classify, 'finalised_model.pkl') 



Analysis()
 

    

