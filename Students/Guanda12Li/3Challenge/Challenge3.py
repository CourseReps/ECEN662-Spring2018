#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:43:33 2018

@author: Andrew
"""

import pandas as pd
import numpy as np

df = pd.DataFrame.from_csv("3challenge-1.csv")
dftraining = df.loc[~np.isnan(df['label'])]
dftesting = df.loc[np.isnan(df['label'])]
print(dftraining.shape)
print(dftesting.shape)

TrainingData = dftraining.as_matrix(columns=None)
TestData = dftesting.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

label = sum(TestData[:,:].T) / 320

for i in range(5000, 10000):
    df['label'][i] = label[i - 5000]

df.to_csv('3challenge_update.csv')