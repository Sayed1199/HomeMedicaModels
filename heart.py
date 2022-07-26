"""
Created on Sat Feb 19 00:10:17 2022
@author: sayed
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

dataset = pd.read_csv('data/heart.csv')
X = dataset.drop(['target'], axis = 1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

classifier=RandomForestClassifier(n_jobs=-1, n_estimators=400,
                                 bootstrap= False,criterion='gini',max_depth=5,
                                 max_features=3,min_samples_leaf= 7)
classifier.fit(X_train, y_train)
print("Accuracy: {}".format(round(accuracy_score(y_test, classifier.predict(X_test))*100,2)))
pickle.dump(classifier, open('heart.pkl', 'wb'))































