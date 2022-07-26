"""
Created on Sat Feb 19 05:47:49 2022
@author: sayed
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
data = pd.read_csv('data/kidneyDisease.csv')
data.classification=data.classification.replace("ckd\t","ckd") 
data.drop('id', axis = 1, inplace = True)
data['classification'] = data['classification'].replace(['ckd','notckd'], [1,0])
df = data.dropna(axis = 0)
df.index = range(0,len(df),1)
df['wc']=df['wc'].replace("\t6200",6200)
df['wc']=df['wc'].replace("\t8400",8400)
df['pcv']=df['pcv'].astype(int)
df['wc']=df['wc'].astype(int)
df['rc']=df['rc'].astype(float)
dictonary = {
        "rbc": {"abnormal":1,"normal": 0,},
        "pc":{"abnormal":1,"normal": 0,},
        "pcc":{"present":1,"notpresent":0,},
        "ba":{"notpresent":0,"present": 1,},
        "htn":{"yes":1,"no": 0,},
        "dm":{"yes":1,"no":0,},
        "cad":{"yes":1,"no": 0,},
        "appet":{"good":1,"poor": 0,},
        "pe":{"yes":1,"no":0,},
        "ane":{"yes":1,"no":0,}}
df=df.replace(dictonary)
X = df.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis = 1)
y = df['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = RandomForestClassifier(n_estimators = 20)
model.fit(X_train, y_train)
print("Accuracy: {}".format(round(accuracy_score(y_test, model.predict(X_test))*100, 2)))
pickle.dump(model, open('kidney.pkl', 'wb'))
























