import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import det

df = pd.DataFrame.from_csv("1challenge_old.csv")
df1 = df.loc[df['label'] == 1.0]
df0 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

AllData = df.as_matrix(columns=['Y0', 'Y1'])

plt.plot(TrainingData0[:, 0], TrainingData0[:, 1], 'x', color='r')
plt.plot(TrainingData1[:, 0], TrainingData1[:, 1], 'x', color='b')
plt.plot(TestData[:, 0], TestData[:, 1], 'o', color='k')
plt.axis('equal')
plt.show()

mean_df1 = np.mean(df1, axis=0)
mean_df1 = np.array([mean_df1[0], mean_df1[1]])

mean_df0 = np.mean(df0, axis=0)
mean_df0 = np.array([mean_df0[0], mean_df0[1]])

cov_df1 = np.cov(df1['Y0'], df1['Y1'])
cov_df0 = np.cov(df0['Y0'], df0['Y1'])

df01_threshold = 6/4

mean_removed_df1 = TestData - mean_df1
mean_removed_df0 = TestData - mean_df0

cov_df1_inv = inv(cov_df1)
cov_df0_inv = inv(cov_df0)

step1_f1 = np.dot(mean_removed_df1,  cov_df1_inv) * mean_removed_df1
step2_f1 = (1/np.sqrt(det(cov_df1_inv)))*np.exp(-0.5*(np.sqrt(np.square(step1_f1[:,0]) + np.square(step1_f1[:,1]))))

step1_f0 = np.dot(mean_removed_df0,  cov_df0_inv) * mean_removed_df0
step2_f0 = (1/np.sqrt(det(cov_df0_inv)))*np.exp(-0.5*(np.sqrt(np.square(step1_f0[:,0]) + np.square(step1_f0[:,1]))))

P_1_over_0 = np.divide(step2_f1, step2_f0)*(1/df01_threshold)

count1s = 0
count0s = 0
for x in range(0,5000):
    if P_1_over_0[x] > 1:
        P_1_over_0[x] = 1
        count1s = count1s + 1
    else:
        P_1_over_0[x] = 0
        count0s = count0s + 1

df.loc[10000:15000,'label'] = P_1_over_0
df.to_csv("1challenge_new.csv")





