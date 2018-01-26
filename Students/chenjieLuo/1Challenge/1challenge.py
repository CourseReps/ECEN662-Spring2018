#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:14:20 2018

@author: Chenjie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

df = pd.DataFrame.from_csv("1challenge.csv")
df0 = df.loc[df['label'] == 1.0]
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)

TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
plt.plot(TestData[:,0], TestData[:,1], 'o', color='k')
plt.axis('equal')
plt.show()

# Here I assume the two Training datas are supposed to follow 2-D Gaussian Distribution
# My initial assumption is taking two columns of data as two independent observations and then the joint pdf can be described as the product of two 1-D Normal Distribution
# However, the result is far from satisfaction, so I take this assumption


mean1 = np.mean(TrainingData0[:,0])
mean2 = np.mean(TrainingData0[:,1])
mean3 = np.mean(TrainingData1[:,0])
mean4 = np.mean(TrainingData1[:,1])

covariance1 = np.cov(TrainingData0[:,0],TrainingData0[:,1])
covariance2 = np.cov(TrainingData1[:,0],TrainingData1[:,1])

y0 = multivariate_normal([mean1,mean2],covariance1)
y1 = multivariate_normal([mean3,mean4],covariance2)
list1 = np.zeros((int(np.size(TestData)/2),1))

def Thresholding(data0,data1,f0,f1):
    ratio = f0.pdf([data0,data1]) / f1.pdf([data0,data1])
    if ratio > 1.5:
        return 1.0
    else:
        return 0.0

for row in TestData:
    Label = Thresholding(row[0],row[1],y0,y1)
    mask = (TestData[:,:] == row)
    list1[mask[:,1]] = Label

Testdata = np.hstack((TestData,list1))
print Testdata
dftest = pd.DataFrame(Testdata,columns=['Y0', 'Y1','label'])


df = pd.concat([df1, df0, dftest], join='outer', ignore_index=True)

# Verification of result
df3 = df.loc[df['label'] == 1.0]
df4 = df.loc[df['label'] == 0.0]

print(df3.shape)
print(df4.shape)

TrainingData0 = df3.as_matrix(columns=None)
TrainingData1 = df4.as_matrix(columns=None)

plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
plt.axis('equal')
plt.show()

df.to_csv("1challenge.csv")