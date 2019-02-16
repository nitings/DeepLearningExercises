
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:23:07 2019

@author: ngs
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding Countries.
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Encoding gender variable
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Creating dummy variables only for country.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Removing the first column to avoid dummy variable trap.
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential # Used to initialize the NN.
from keras.layers import Dense # To create layers.
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    #Initializing the ANN.
    classifier = Sequential()
    
    # Lets apply rectification actv. function for hidden layers and
    # Sigmoid for outer layers.
    # Here units is number of nodes in the layer.
    classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))
    #Adding another hidden layer.
    classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation='relu'))
    classifier.add(Dropout(p=0.1))
    #Adding the output layer.
    classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X=X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()