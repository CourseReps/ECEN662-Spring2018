#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:48:34 2018

@author: Chenjie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame.from_csv("3challenge-1.csv")
dftraining = df.loc[~np.isnan(df['label'])]
dftesting = df.loc[np.isnan(df['label'])]
print(dftraining.shape)
print(dftesting.shape)
            

TrainingData = dftraining.as_matrix(columns=None)
TestData = dftesting.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])
print(TestData.shape)

theta_hat = np.sum(TestData,axis = 1) / (40 * 8)
theta_hat = theta_hat.reshape((5000,1))

Result = np.hstack((dftesting.as_matrix(columns = ['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7']),theta_hat))
dftesting = pd.DataFrame(Result,columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'])

df = pd.concat([dftraining, dftesting], join='outer', ignore_index=True)
df.to_csv("3challenge-1.csv")
