#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:29:47 2018

@author: Humberto
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
#Read the csv file
df = pd.DataFrame.from_csv("2challenge_backup.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)

#Convert data into numpy arrays
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7'])

#Estimate the mean and covariance for the training data.
#First compute mean given label 0
#
##We find mean and covariance for labels 0.
means0 = np.nanmean(TrainingData0[:,0:7],axis = 0)
print 'The means for labels 0 are \n', means0
stds0 = np.nanstd(TrainingData0[:,0:7],axis = 0)
print 'The stds for labels 0 are \n ' , stds0
m0 = np.mean(means0)

##We find mean and covariance for labels 1.
means1 = np.nanmean(TrainingData1[:,0:7],axis = 0)
print 'The means for labels 1 are \n', means1
stds1 = np.nanstd(TrainingData1[:,0:7],axis = 0)
print 'The stds for labels 1 are \n ' , stds1
m1 = np.mean(means1)

#From here we can see that it is reasonable to consider the observations 
#are assume to be Gaussian and identically distributed.

#Also, the priors are taken to be equal.
Pr0 = 0.5
Pr1 = 0.5
#And we can compute the threshold tau.
tau = 0.5*(m1 + m0)

#Then, according to the observations, we can compute the
#likelihood ratio as
count1 = 0.0
count0 = 0.0
#The variable Data is just a place holder for the data we want to classify.
Data = TestData
labels = np.zeros(len(Data[:,0]))
#    Check how many nan are present in the row.
#     numOfNans = np.count_nonzero(np.isnan(Data[k,:]))>0
#     if numOfNans > 0 :
#         num_observations = 7- numOfNans
#Take the mean for every sequence, ignoring nans and ignoring label.
yi = np.nanmean(Data[:,0:7], axis = 1)
print yi
#Classification according to threshold
for k in range(0,yi.shape[0]):
    if yi[k] >= tau:
        labels[k] = 1.0
        count1+= 1
    else:
        labels[k] = 0.0
        count0+= 1
            
TestData_labeled = np.c_[Data,labels]

dftest = pd.DataFrame(TestData_labeled,columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'] )

#Data visualization
print("Total of 0 is ",count0)
print("Total of 1 is ",count1)
print("Zeros percentage",count0/(count0+count1))
tk = np.arange(7)

DataToPlot = TestData_labeled
for k in range (0,len(Data[:,0])):
    if DataToPlot[k,8] == 0:
        plt.plot(tk, DataToPlot[k,0:7], 'x', color='r')
    else:
        plt.plot(tk, DataToPlot[k,0:7], 'o', color='b', alpha = 0.05)
    
plt.axis('equal')
plt.grid()
plt.show()

##Write the csv
df = pd.concat([df0, df1, dftest], join='outer', ignore_index=True)
df.to_csv("2challenge.csv")
