#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:45:10 2018

@author: Andrew
"""

'''
some trainingdatas have 6 features and some have 8 features. So i will make two classifier, one based on 6 
features, and another one based on 8 features. when classify the testdata, we will use classifier based on 
number of features it has.

exhaustive search
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from itertools import combinations
    
df = pd.DataFrame.from_csv("2challenge.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]


TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=None)

#plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
#plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
#plt.plot(TestData[:,0], TestData[:,1], 'o', color='k')
#plt.axis('equal')
#plt.show()
'''
feature selection （exhaustive search QDA)
pick best 6 features. However, the result between top 6 features and Y0-Y5 six features is very close.
'''
def filterdata(TrainingData):
    filter = TrainingData
    i = 0
    ct = 0
    while i < len(TrainingData):
        if np.isnan(TrainingData[i, 6]):
            filter = np.delete(filter, i - ct, axis = 0)
            ct +=1
        i +=1
    return filter

def QDA_score(X, Y, i):
    feature_array = np.asarray(X)
    feature_array.shape = (5004, i)
    clf_QDA = QDA()
    clf_QDA.fit(X, Y)
    return clf_QDA.score(X, Y)

def transform(sample_col):
    row = len(sample_col)
    col = len(sample_col[0])
    return [[sample_col[i][j] for i in range(row)] for j in range(col)]


filter_0 = filterdata(TrainingData0)
filter_1 = filterdata(TrainingData1)

X = np.concatenate((filter_0[:, 0:8], filter_1[:, 0:8]), axis = 0)       
Y = np.concatenate((filter_0[:, 8], filter_1[:, 8]), axis = 0)

exhaus_QDA_dic = {}
X_T = transform(X)
for i in range(1, 9):
    y = combinations(range(8), i)
    for j in y:
        fea_exhaustive = []
        exhaus_list = []
        for k in j:
            fea_exhaustive.append(X_T[k])            
            exhaus_list.append(k)
            exhaus_tp = tuple(exhaus_list)
        fea_exhaustive = transform(fea_exhaustive)
        exhaus_QDA_dic[exhaus_tp] = QDA_score(fea_exhaustive, Y, i)

'''
find the best combination
'''
best_QDA_exhaustive = {}
for i in range(1, 9):
    max_QDA_header = None
    max_QDA = None
    for a, b in exhaus_QDA_dic.items():
        if len(a) == i:
            if max_QDA == None or b > max_QDA:
                max_QDA_header = a
                max_QDA = b
    best_QDA_exhaustive[max_QDA_header] = max_QDA

'''
pick the first 6 features can get a pretty good result
'''
final_X = np.concatenate((TrainingData0[:, 0:6], TrainingData1[:, 0:6]), axis = 0)       
final_Y = np.concatenate((TrainingData0[:, 8], TrainingData1[:, 8]), axis = 0)
final_test = TestData[:, 0:6]


final_clf_QDA = QDA()
final_clf_QDA.fit(final_X, final_Y)
result_QDA = final_clf_QDA.predict(final_test)
apparent_error_QDA = 1 - final_clf_QDA.score(final_X, final_Y)
QDA_label = np.zeros((5000,1))
QDA_label[:,0]= final_clf_QDA.predict(final_test)

Predict_data = np.concatenate((final_test, QDA_label), axis = 1)
All_data = np.concatenate((np.column_stack((final_X, final_Y)), Predict_data[:, 0:7]), axis = 0)
'''
save the data
'''
df_new=pd.DataFrame({'Y0':All_data[:,0],'Y1':All_data[:,1], 'Y2':All_data[:,2], \
                     'Y3':All_data[:,3], 'Y4':All_data[:,4], 'Y5':All_data[:,5],'label':All_data[:,6]})
df_new.to_csv("2challenge_update.csv")
        
        

