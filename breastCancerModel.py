"""
Created on Sat Feb 19 01:25:36 2022
@author: sayed
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import pickle
from scipy.stats import randint

dataset = pd.read_csv('data/breastCancer.csv')
dataset.drop(['id','symmetry_se','smoothness_se','texture_se','fractal_dimension_mean'], axis = 1, inplace = True)
dataset['diagnosis'].replace(['M','B'], [1,0], inplace = True)
dataset.drop('Unnamed: 32',axis = 1, inplace = True)
X = dataset.drop('diagnosis', axis = 1)
y = dataset['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
RFmodel = RandomForestClassifier(n_estimators=20)
RFmodel.fit(X_train, y_train)
classifier = RandomForestClassifier(n_jobs = -1)
param_dist={
    'max_depth':[3,5,10,None],'n_estimators':[10,100,200,300,400,500],'max_features':randint(1,27),
               'criterion':['gini','entropy'],'bootstrap':[True,False], 'min_samples_leaf':randint(1,27),
              }
search_clfr = RandomizedSearchCV(classifier, param_distributions = param_dist, n_jobs=-1, n_iter = 40, cv = 9)
search_clfr.fit(X_train, y_train)
params = search_clfr.best_params_
score = search_clfr.best_score_
classifier=RandomForestClassifier(n_jobs=-1, n_estimators=200,bootstrap= True,criterion='gini',max_depth=20,max_features=8,min_samples_leaf= 1)
classifier.fit(X_train, y_train)
print("RandomForest Accuracy: {}".format(round(accuracy_score(y_test, RFmodel.predict(X_test))*100,2)))
pickle.dump(classifier, open('Models/breastCancer.pkl', 'wb'))













