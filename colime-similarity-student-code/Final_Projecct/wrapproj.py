import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.metrics import jaccard_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE


import seaborn
import matplotlib

from pathlib import Path
import glob
import logging  


from sklearn.preprocessing import KBinsDiscretizer

##########
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.utils.vis_utils import plot_model

from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from keras import models    
from tensorflow.keras.utils import plot_model


#from keras.layers import Dense


from tensorflow.keras.models import load_model

from IPython.display import SVG
 
from sklearn.model_selection import KFold 
KFold

#########





############## For all the graphs
plt.rcParams['figure.figsize'] = (14,6)
#############
#

#
######## READ THE DATA #########
#for name in glob.glob('data/*.*'):
#    print(name)

heart = pd.read_csv('data/heart.csv.xls', index_col=0)
df = heart.copy()


###  For plotting ##

def print_evaluations(ytrue, ypred, model):
    print(f'How does model {model} score:')
    print(f'The accuracy of the model is: {round(accuracy_score(ytrue, ypred), 3)}')
    print(f'The precision of the model is: {round(precision_score(ytrue, ypred), 3)}')
    print(f'The recall of the model is: {round(recall_score(ytrue, ypred), 3)}')
    print(f'The f1-score of the model is: {round(f1_score(ytrue, ypred), 3)}')
    
    #print confusion matrix
    fig = plt.figure(figsize=(6, 6))
    cm = confusion_matrix(ytrue, ypred)
    print(cm)
    
    #plot the heatmap
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['no heart diseases', 'heart diseases']); 
    ax.yaxis.set_ticklabels(['no heart diseases', 'heart diseases'])
    return


###########  Split the data  ########

X = df.iloc[:,:-1] #all rows (:), and all columns EXCEPT for the last one
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)



########Logistic regression ############
def LogRegfunc(X, y):
        # normalization of the datset
	X = preprocessing.StandardScaler().fit(X).transform(X)
  
	# Train-and-Test -Split
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 4)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)

	mlg = LogisticRegression()
	mlg.fit(X_train, y_train)
	y_pred = mlg.predict(X_test)
	clg = mlg.coef_, mlg.intercept_ 
	return  y_test, y_pred, clg

y_test, y_pred, clg  = LogRegfunc(X, y)
print_evaluations(y_test, y_pred, 'Logistic Regression')

#print('Accuracy of the model in jaccard similarity score is = ', jaccard_score(y_test, y_pred))
#print('Coeficients', clg)

print('')
print('###############')
print('')

####Decision tree classifier######

def  DecisionTreeCfun(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)
	mtr = DecisionTreeClassifier(max_depth=3)   #Hyperparameters -> there are many more you can try out against overfitting
	mtr.fit(X_train, y_train)
	y_pred = mtr.predict(X_train)   # prediction
	mtr.score(X_train, y_train)
	return y_test, y_pred
y_test, y_pred = DecisionTreeCfun(X,y)  

print_evaluations(y_train, y_pred, 'Decision tree classifier ')




print('')
print('###############')
print('')

def baseline(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)
	ypred_baseline = [1] * len(y_train.values)
	return ypred_baseline,y_train  
ypred_baseline,y_train = baseline(X,y)
#generating predictions of all ones, i.e. the model guesses  DISEASES  for all cases.
print_evaluations(y_train, ypred_baseline, 'Baseline Model')




print('')
print('###############')
print('')



def RandomForestfunc(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)
	rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=10)
	rf.fit(X_train, y_train)
	ypred_rf = rf.predict(X_test)
	return ypred_rf, y_test
ypred_rf, y_test  = RandomForestfunc(X,y)
print_evaluations(y_test, ypred_rf, 'RandomForest')




print('')
print('###############')
print('')



#######   Methods to improve: Use Undersampling  #####


def  RanSamplerfunc(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)
	print((y_train == 0).sum(), (y_train == 1).sum())
	rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=10)

    
	rus = RandomUnderSampler(sampling_strategy={0:95},random_state=10) 
	#This is  modified depending on the number of samples
	# we are asking for 95 data points out of 103 in the first class. ## (y_train == 0).sum(), (y_train == 1).sum()

	nm = NearMiss(sampling_strategy={0: 95}) ####  This is modified depending on the number of samples.
						####(y_train == 0).sum(), (y_train == 1).sum()


	X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train) 
	#fit_resample() new imblearn syntax
	#very conceptually similar to .fit_transform() <---sklearn

	X_train_nm, y_train_nm = nm.fit_resample(X_train, y_train) 
 

	### Exact same code as before, but this time we are training the Random Forest on the undersampled  / down-sampled
	rf.fit(X_train_rus, y_train_rus)
	ypred_rus = rf.predict(X_test)


	return ypred_rus, y_test, X_train_nm, y_train_nm

         


ypred_rus, y_test, X_train_nm, y_train_nm  =RanSamplerfunc(X,y)
print_evaluations(y_test, ypred_rus, 'Random Undersampling')




print('')
print('###############')
print('')

#######  Near Miss ############

rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=10)
rf.fit(X_train_nm, y_train_nm)
ypred_nm = rf.predict(X_test)
print_evaluations(y_test, ypred_nm, 'Near Miss')



print('')
print('###############')
print('')

#######  Over samplimng  SMOTE ############


def  OverSamplerfunc(X,y,rf):
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)
	print((y_train == 0).sum(), (y_train == 1).sum())
	ros= RandomOverSampler(random_state=10,sampling_strategy={1:520})
	#up-sampling--- the minority class to have 520  instead
 
	X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
	# Exact same code as before, but this time we are training 
	#cthe Random Forest on the undersampled  / down-sampled data using Near Miss

	rf.fit(X_train_ros, y_train_ros)
	ypred_ros = rf.predict(X_test)
	return ypred_ros, y_test
ypred_ros, y_test = OverSamplerfunc(X,y,rf)
print_evaluations(y_test, ypred_ros, 'Random Oversampling')



print('')
print('###############')
print('')


#X = df.iloc[:,:-1] #all rows (:), and all columns EXCEPT for the last one
#y = df['output']


######### Artificial Nueral Network (ANN) ####

def     ANN_func(X,y):  
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)


	# keras model



	classifier = Sequential()
	classifier.add(Dense(activation = "elu", input_dim = 12, units = 8, kernel_initializer = "uniform"))
	classifier.add(Dense(activation = "elu", units = 13, kernel_initializer = "uniform"))
	classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))
	classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )

	hist= classifier.fit(X_train , y_train , batch_size = 10, epochs =200, validation_split=0.2)

	classifier.summary()
	classifier.evaluate(X_train,y_train)
	classifier.predict(X_train)

	y_pred = classifier.predict(X_test)
	y_pred = (y_pred > 0.5)

	return y_test, y_pred

y_test, y_pred  = ANN_func(X,y)

print_evaluations(y_test, y_pred, 'ANN')	



