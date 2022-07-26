"""
Created on Sat Feb 19 01:25:36 2022

@author: sayed
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle

data = pd.read_csv('data/diabetes.csv')
print("Shape: ",data.shape)
print("Null Sum: \n",data.isnull().sum())
print("Corr: \n",data.corr)

X = data.iloc[:,:-1]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 10)
######### Decision Tree Classification  #############
modelDCT =  DecisionTreeClassifier()
modelDCT.fit(X_train,y_train)
modelDCTPrediction = modelDCT.predict(X_test)
print("DecisionTree Accuracy: ",accuracy_score(y_test, modelDCTPrediction)*100)
print(classification_report(y_test,modelDCTPrediction))

######### RandomForrest Classification  #############
modelRFST =  RandomForestClassifier(n_jobs=-1, n_estimators=400,
                                 bootstrap= False,criterion='gini',max_depth=5,
                                 max_features=3,min_samples_leaf= 7)
modelRFST.fit(X_train,y_train)
modelRFSTPrediction = modelRFST.predict(X_test)
print("RandomForrest Accuracy: ",accuracy_score(y_test, modelRFSTPrediction)*100)
print(classification_report(y_test,modelRFSTPrediction))

######### SKFOLD Classification  #############
skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)
modelSKFold = DecisionTreeClassifier()
resultSKFold = cross_val_score(modelSKFold, X, y, cv=skfold)
# holdout validation
result = modelDCT.score(X_train,y_train)
print("Accuracy: %.2f%%" %(result*100.0))
pickle.dump(modelDCT, open("diabetes.pkl",'wb'))









