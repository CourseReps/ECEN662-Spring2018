#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:38:13 2018

@author: santosh
"""

import pandas as pd
import numpy as np
from numpy import matrix
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
import scipy as sp
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame.from_csv('2challenge.csv')
df5=df.loc[pd.isnull(df['Y6'])]
df7=df.loc[~(pd.isnull(df['Y6']))]
df7_train= df7.loc[((df7['label'] == 0.0) | (df7['label'] == 1.0))]
df7_test= df7.loc[~((df7['label'] == 0.0) | (df7['label'] == 1.0))]
df5_train= df5.loc[((df5['label'] == 0.0) | (df5['label'] == 1.0))]
df5_test= df5.loc[~((df5['label'] == 0.0) | (df5['label'] == 1.0))]
print(df7.shape)
print(df5.shape)
print(df7_test.shape)
TrainingData5 = df5_train.as_matrix(columns=None)
X5=matrix(TrainingData5[:,:6])
y5=np.transpose(matrix(TrainingData5[:,8]))
TrainingData7 = df7_train.as_matrix(columns=None)
X7=matrix(TrainingData7[:,:8])
y7=np.transpose(matrix(TrainingData7[:,8]))
TestData5 = df5_test.as_matrix(columns=None)
X5_t=matrix(TestData5[:,:6])
y5_t=np.transpose(matrix(TestData5[:,8]))
TestData7 = df7_test.as_matrix(columns=None)
X7_t=matrix(TestData7[:,:8])
y7_t=np.transpose(matrix(TestData7[:,8]))
#TrainingData1 = df1.as_matrix(columns=None)
#TrainingData  = df2.as_matrix(columns=None)

#plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
#plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
#plt.plot(TestData[:,0], TestData[:,1], 'o', color='k')
#plt.axis('equal')
#x=matrix([np.ones(10000),TrainingData[:,0],TrainingData[:,1]])
#x=np.transpose(x)
#x1=matrix([TrainingData[:,0],TrainingData[:,1]])
#x1=np.transpose(x1)
#theta=matrix([0,0,0])
#theta=np.transpose(theta)
#y=matrix([TrainingData[:,2]])
#y=np.transpose(y)

sc = StandardScaler()
sc.fit(X5)
X5 = sc.transform(X5)
from sklearn.linear_model import LogisticRegression
logit1=LogisticRegression(C=1.0, random_state=0)
logit1.fit(X5,y5)    
logit1.score(X5,y5)
y5_t=np.transpose(matrix(logit1.predict(X5_t)))
a1=matrix(df5_test.index)

for m in range(0, 2452):
  df['label'][a1[0,m]]=y5_t[m]

sc.fit(X7)
X7 = sc.transform(X7)
from sklearn.linear_model import LogisticRegression
logit2=LogisticRegression(C=1.0, random_state=0)
logit2.fit(X7,y7)    
logit2.score(X7,y7)
y7_t=np.transpose(matrix(logit2.predict(X7_t)))
a=matrix(df7_test.index)

for m in range(0, 2548):
  df['label'][a[0,m]]=y7_t[m]
  
df.to_csv('2challenge1.csv')
