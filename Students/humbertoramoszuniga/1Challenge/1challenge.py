import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
#Read the csv file
df = pd.DataFrame.from_csv("1challenge.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)


#Convert data into numpy arrays
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])


#Estimate the mean and covariance for the training data.
#First compute mean given label 0
#Fit the training data to a 2D gaussian distribution.

#We find mean and covariance for the data that was mapped from a 0.
mu_given00,std_dev_given00 = norm.fit(TrainingData0[:,0])
print(mu_given00,std_dev_given00)


mu_given01,std_dev_given01 = norm.fit(TrainingData0[:,1])
print(mu_given01,std_dev_given01)

#Fit the training data to a 2D gaussian distribution.
#We find mean and covariance for the data that was mapped from a 1.
mu_given10,std_dev_given10 = norm.fit(TrainingData1[:,0])
print(mu_given10,std_dev_given10)


mu_given11,std_dev_given11 = norm.fit(TrainingData1[:,1])
print(mu_given11,std_dev_given11)

cov_given0 = np.cov(TrainingData0[:,0:2],rowvar = False)
cov_given1 = np.cov(TrainingData1[:,0:2],rowvar = False)

#Given the results, it is not a bad assumption that the data is corrupted by a
#gaussian process with zero mean and unit covariance for training0 data and
#correlated data as given by cov_given1.
#Also, the priors are taken to be in proportion to the known training data.
#So,
Pr0 = 0.6
Pr1 = 0.4
#And we can compute the threshold tau as
tau = Pr0/Pr1

#Now with the distributions (mean and covariances), we can compute the
#likelihood ratio and threshold.
#Let's compute the likelihood ratio.
f0 = multivariate_normal([mu_given00, mu_given01], cov_given0)
f1 = multivariate_normal([mu_given10, mu_given11], cov_given1)
#
count1 = 0.0
count0 = 0.0
labels = np.zeros(len(TestData[:,0]))

Data = TestData
for k in range(0,Data.shape[0]):
#    print("k is ",k)
    LR = f1.pdf(Data[k,0:2])/f0.pdf(Data[k,0:2])
#    print("LR is ",LR)
#    LR = f1.pdf(TrainingData0[1,0:1])/f0.pdf(TrainingData0[1,0:1])
#Classification according to threshold

    if LR >= tau:
        count1+= 1
        labels[k] = 1.0
#        print(1)
    else:
#        print(0)
        count0+=1
TestData_labeled = np.c_[TestData,labels]
        
dftest = pd.DataFrame(TestData_labeled,columns = ['Y0','Y1','label'] )        
        
#Data visualization
print("Total of 0 is ",count0)
print("Total of 1 is ",count1)
#print("Zeros perc",count0/(count0+count1))

for k in range (0,len(TestData[:,0])):
    if TestData_labeled[k,2] == 0:
        plt.plot(TestData_labeled[k,0], TestData_labeled[k,1], 'x', color='r')
    else:
        plt.plot(TestData_labeled[k,0], TestData_labeled[k,1], 'o', color='b')
plt.axis('equal')
plt.show()

#Write the csv
df = pd.concat([df0, df1, dftest], join='outer', ignore_index=True)
df.to_csv("1challenge.csv")
