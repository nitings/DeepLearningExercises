#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:05:12 2019

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

# Part2: Making the ANN.
import keras
from keras.models import Sequential # Used to initialize the NN.
from keras.layers import Dense # To create layers.

#Initializing the ANN.
classifier = Sequential()

# Lets apply rectification actv. function for hidden layers and
# Sigmoid for outer layers.
# Here units is number of nodes in the layer.
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation='relu', input_dim=11))

#Adding another hidden layer.
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation='relu'))

#Adding the output layer.
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation='sigmoid'))

#Compiling the ANN. Means applying stochastic gradient descent on our ANN.
# Adam is a stochastic gradient descent implementation.
# loss is similar to the cost function used in logistic regression.
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to training set.
# Batchsize the size after which we update the weights.
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Predict method gives probability, we need to convert it into Yes or No.
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

