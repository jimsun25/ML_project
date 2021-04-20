# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:39:03 2019

@author: jimsu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

# Feature Scaling(Decision Tree is not based on Euclidean distance, so can be removed, but for higher resolusion plotting, keep for now.)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state =1 )
classifier.fit(X_train, y_train)

# Predicting the Test set Result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
 np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
X1_ravel = X1.ravel()
X2_ravel = X2.ravel()
X1X2_array = np.array([X1_ravel, X2_ravel])
X1X2_array_t = X1X2_array.T
X1X2_pred = classifier.predict(X1X2_array_t)
X1X2_pred_reshape = X1X2_pred.reshape(X1.shape)
result_plt = plt.contourf(X1, X2, X1X2_pred_reshape, alpha=0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = [ListedColormap(('red', 'green'))(i)], label = j)
plt.title('Random Forest (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualizing the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
 np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
X1_ravel = X1.ravel()
X2_ravel = X2.ravel()
X1X2_array = np.array([X1_ravel, X2_ravel])
X1X2_array_t = X1X2_array.T
X1X2_pred = classifier.predict(X1X2_array_t)
X1X2_pred_reshape = X1X2_pred.reshape(X1.shape)
result_plt = plt.contourf(X1, X2, X1X2_pred_reshape, alpha=0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = [ListedColormap(('red', 'green'))(i)], label = j)
plt.title('Random Forest (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
