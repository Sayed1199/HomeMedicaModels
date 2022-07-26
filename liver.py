"""
Created on Sat Feb 19 06:53:12 2022
@author: sayed
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


data = pd.read_csv('data/liver.csv')

data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(0.947064)

data['Dataset'] = data['Dataset'].replace([2,1],[1,0])

data = pd.get_dummies(data, columns = ['Gender'], drop_first = True)

X = data.drop('Dataset', axis = 1)
y = data['Dataset']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)


model = RandomForestClassifier(n_estimators=20)

model.fit(X_train, y_train)

print("Accuracy: {}".format(round(accuracy_score(y_test, model.predict(X_test))*100,2)))

pickle.dump(model, open('liver.pkl', 'wb'))














