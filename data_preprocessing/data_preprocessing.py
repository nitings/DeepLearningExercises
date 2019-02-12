# Data Preprocessing.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:34:30 2019

@author: ngs
"""

# Importing the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset.
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values

Y = dataset.iloc[:, -1].values

# Addressing the missing data.
# Replace the missing data with the mean of that column.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean",
                  axis=0)
imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

# Now need to encode categorical variables into number.
# Since countries name can't be put into mathematical equations.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#Just changing them to numbers wont help as algorithms may
# rate one country above other based on the value.
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# For dependent variable we will only use label encoder.
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Now we need to split the dataset into training set and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
