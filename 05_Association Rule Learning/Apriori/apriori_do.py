# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:24:10 2019

@author: jimsu
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and making a list of lists
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_lengh = 2)

# Visualizing the results
results = list(rules)
output = []
for row in results:
    output.append(str(row.items))