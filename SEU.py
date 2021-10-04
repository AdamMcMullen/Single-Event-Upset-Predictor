# Based on the tutorial here https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

import sys
import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from Data import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

"""
Author: Adam McMullen
Team: Team NorthStar
Motto: Sic Itur Ad Astra
Date: October 3, 2021

This script passes the CASSIOPE SEU data through various machine learning algorithms to develop a program that can
predict orbital parameters and space weather that will have a risk of SEUs.
"""

dataset = dataset[[' Latitude (deg)',' Longitude (deg)',' Altitude (km)','Year','LocalTime','kp','f10','dst','SAA','class']]
test = test[[' Latitude (deg)',' Longitude (deg)',' Altitude (km)','Year','LocalTime','kp','f10','dst','SAA']]
# Split-out validation dataset
array = dataset.values
X = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=0)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make predictions on test dataset
predictions = model.predict(test.values)
print(predictions)