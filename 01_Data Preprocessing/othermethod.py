# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:39:47 2019

@author: jimsu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3:].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer(transformers = [('transformer', OneHotEncoder(), [0])], remainder = 'passthrough')
X = transformer.fit_transform(X)
Y = transformer.fit_transform(Y)