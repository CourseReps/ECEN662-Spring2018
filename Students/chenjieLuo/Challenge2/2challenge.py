#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed feb 13 11:14:20 2018

@author: Chenjie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, neighbors
from sklearn.neighbors import KNeighborsClassifier

#Since the number of 0's and 1's are equal, therefore, the first algorithm occurs in my mind is KNN algorithm. KNN basically detects nearest k number
#of points around the point i and compare their labels to determine point i's label

df = pd.DataFrame.from_csv("2challenge.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
df2 = df.loc[(df['label'] == 0.0) | (df['label'] == 1.0)]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

Demo0 = df0.as_matrix(columns = ['Y1','Y3'])
Demo1 = df1.as_matrix(columns = ['Y1','Y3'])
Demox = dftest.as_matrix(columns = ['Y1','Y3'])

TrainingData = df2.as_matrix(columns = ['Y0','Y1','Y2','Y3','Y4','Y5'])
TrainingClass = np.array(df2['label'])


#After analyzing the data point projection on any 2-D space, we can say actually the threshold
#is intuitively obvious,and hence I decided abandon last two columns of data to analyze since
#we miss certain points' Y6 and Y7's value. Below I use Y1 and Y3 as an example

plt.plot(Demo0[:,0], Demo0[:,1], 'x', color='r')
plt.plot(Demo1[:,0], Demo1[:,1], 'x', color='b')
plt.plot(Demox[:,0], Demox[:,1], 'o', color='k')
plt.axis('equal')
plt.show()
Undefined_Data = dftest.as_matrix(columns = ['Y0','Y1','Y2','Y3','Y4','Y5'])
Undefined_Class = np.array(dftest['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(TrainingData, TrainingClass, test_size = 0.2)
clf_test = KNeighborsClassifier(9)
clf_test.fit(x_train,y_train)
accuracy = clf_test.score(x_train,y_train)
print('The accuracy of the y_test based on trainging set is: ')
print(accuracy)

clf = KNeighborsClassifier(9)
clf.fit(TrainingData,TrainingClass)
Undefined_Class = clf.predict(Undefined_Data)
Undefined_Class = Undefined_Class.reshape((5000,1))

ResultData = np.hstack((dftest.as_matrix(columns = ['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7']),Undefined_Class))
dftest = pd.DataFrame(ResultData,columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'])

df = pd.concat([df2, dftest], join='outer', ignore_index=True)

df.to_csv("2challenge.csv")