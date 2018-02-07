import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.DataFrame.from_csv("1challenge.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
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

'''
Self Assumptions:
1, Pr(0)/Pr(1) shares the same ratio as the amount of the samples
2, The two distributions are all 2-dimension Gaussian distributions
3, Use Most Likely method to estimate the mu and Sigma of the distribution
'''
#estimate the mean and u of the 2-dim Guassian
def  estimate(X):
      u = [np.mean(X[:,0]),np.mean(X[:,1])]
      A =  np.dot((X[:,0:2] - u).T,(X[:,0:2] - u))
      Sigma = A*3/np.size(X)
      return u,Sigma

def determine(X,mu1,Sigma1,mu2,Sigma2,threshold):
    f1 = 1/(2 * np.pi * np.sqrt(np.linalg.det(Sigma1))) * np.exp(-1/2  * np.dot(np.dot((X-mu1).T, np.power(Sigma1,-1)),(X - mu1)))
    f2 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma2))) * np.exp(-1/2 * np.dot(np.dot((X-mu2).T, np.power(Sigma2,-1)),(X - mu2)))
    ratio = f2/f1;
    if ratio > threshold:
        d = 1
    else:
        d = 0
    return d

u1,Sigma1 = estimate(TrainingData1)
u0,Sigma0 = estimate(TrainingData0)

threshold = 1.5

if TestData.size != 0:
    decision = np.zeros((int(np.size(TestData)/2),1));

    for X in TestData:
       d = determine(X,u0,Sigma0,u1,Sigma1,threshold)
       mask = (TestData[:,:] == X)
       t = mask[:,1]
       decision[mask[:,1]] = d

    TestData1 = np.hstack((TestData,decision))
    dftest = pd.DataFrame(TestData1,columns=['Y0', 'Y1','label'])



df = pd.concat([df0, df1, dftest], join='outer', ignore_index=True)
df.to_csv("1challenge.csv")