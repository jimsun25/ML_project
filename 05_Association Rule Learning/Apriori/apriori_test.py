# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:17:16 2019

@author: jimsu
"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset=pd.read_csv('Market_Basket_Optimisation.csv', header=None)



transactions=[]

for i in range (0,7501):

  transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#transactions = dataset.values.tolist()



#delete the nan objects

for i in transactions:

    for x in range (0,i.count('nan')):

      i.remove('nan')



#training Apriori on the dataset

from apyori import apriori

rules=apriori(transactions, min_support=0.003,min_confidence=0.2, min_lift=3, min_length=2)

   

#visualising the results

results=list(rules)

final_results=pd.DataFrame(columns=['Products', 'Likely', 'Support', 'Confidence', 'Lift'])

for i in results:

  final_results=final_results.append({'Products':list(i.ordered_statistics[0].items_base),

                                      'Likely':list(i.ordered_statistics[0].items_add),

                                      'Support':i[1],

                                      'Confidence':i.ordered_statistics[0].confidence,

                                      'Lift':i.ordered_statistics[0].lift,},

                                        ignore_index=True,)



#sorting by the lift 

final_results=final_results.sort_values(by='Lift', ascending=False)