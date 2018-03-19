import sklearn.svm as svm
import pandas as pd
import numpy as np

clf = svm.SVC()

df = pd.DataFrame.from_csv("2challenge.csv")
dftrain = df.loc[((df['label'] == 0.0) | (df['label'] == 1.0))]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

TrainingData = dftrain.as_matrix(columns=['Y0', 'Y1','Y2','Y3','Y4','Y5'])
TestData = dftest.as_matrix(columns=['Y0', 'Y1','Y2','Y3','Y4','Y5'])

TrainingLabel = dftrain.as_matrix(columns=['label'])

clf.fit(TrainingData,TrainingLabel)
cro_test = clf.predict(TrainingData)
pre = clf.predict(TestData)
pre = pre.reshape((5000,1))
error = np.sum(np.abs(TrainingLabel - cro_test.reshape((10000,1))))/10000

TestData1 = np.hstack((dftest.as_matrix(columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7']),pre))
dftest = pd.DataFrame(TestData1,columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'])



df = pd.concat([dftrain, dftest], join='outer', ignore_index=True)
df.to_csv("2challenge_after.csv")