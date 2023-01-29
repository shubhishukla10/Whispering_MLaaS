import numpy as np
import pandas as pd
import pickle
import ctypes
import pathlib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns


data=pd.read_csv('Attack_Dataset/MLP_training_data.csv')


X = data.loc[:, data.columns != 'Class']
y=data['Class']
X, y = shuffle(X, y)

scaler = StandardScaler()
scaler.fit(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2, shuffle=True)



GRID = [
    {'scaler': [StandardScaler()],
     'estimator': [MLPClassifier(random_state=2)],
     'estimator__solver': ['adam'],
     'estimator__learning_rate_init': [0.001],
     'estimator__max_iter': [2000],
     'estimator__hidden_layer_sizes': [(50,50,50), (400, 400, 400,50), (300, 300, 300,300), (200, 200, 200), (350,200,100, 50)],
     'estimator__activation': ['relu'],
     'estimator__alpha': [0.0001, 0.001, 0.005],
     'estimator__early_stopping': [True, False]
     }
]

PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])

clf = GridSearchCV(estimator=PIPELINE, param_grid=GRID, 
                            scoring=make_scorer(accuracy_score),# average='macro'), 
                            n_jobs=-1, refit=True, verbose=1, 
                            return_train_score=False)

clf.fit(X_train,y_train)

print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)

ypred=clf.predict(X_test)

print("MLP Attack Accuracy and Confusion Matrix for validation data:")
print(accuracy_score(y_test,ypred))

cm = confusion_matrix(ypred, y_test)
print(cm)


#Test MLP on test data
data=pd.read_csv('Attack_Dataset/MLP_test_data.csv')

X = data.loc[:, data.columns != 'Class']
y=data['Class']
X, y = shuffle(X, y)

scaler = StandardScaler()
scaler.fit(X)

ypred=clf.predict(X)

print("MLP Attack Accuracy and Confusion Matrix for test data:")
print(accuracy_score(y,ypred))

cm = confusion_matrix(ypred, y)
print(cm)
