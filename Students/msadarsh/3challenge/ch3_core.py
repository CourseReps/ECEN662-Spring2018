import pandas as pd
import numpy as np

df = pd.DataFrame.from_csv("3challenge-1.csv")
dftraining = df.loc[~np.isnan(df['label'])]
dftesting = df.loc[np.isnan(df['label'])]
print(dftraining.shape)
print(dftesting.shape)


TrainingData = dftraining.as_matrix(columns=None)
TestData = dftesting.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

# print("Training data")
# print(TrainingData[0])
# print(TrainingData[1])

# print("Test Data")
# print(TestData[0])
# print(TestData[1])

# estimating training data

# print(TrainingData[0])

sqsum=0
count=0
for i in range(TrainingData.shape[0]):
# for i in range(1,2):    
    count=count+1
    val=0
    for j in range(TrainingData.shape[1]-1):
        val=val+TrainingData[i][j]
    val=val/((TrainingData[i].size)-1)
    estimate=(val+2)/47
    diff=(estimate-TrainingData[i][8]) 
    sqsum=sqsum+(diff*diff)
    # print("sqsum = %f" % sqsum)
    # print("val = %f" % val)
    # print("estimate = %f" % estimate)
    # print("TrainingData = %f" % TrainingData[i][8])   

sqsum=sqsum/5000
print(count)
print(sqsum)    

countv=0
estimate_ar=[]
print("on test data")
print(TestData.shape)
for i in range(TestData.shape[0]):   
    # countv=countv+1
    val=0
    for j in range(TestData.shape[1]):  
        val=val+TestData[i][j]

    val=val/(TestData[i].size)

    estimate=(val+2)/47 # E[Theta|Y=val]
    estimate_ar=estimate_ar+[estimate]

# print(estimate_ar)

estimate_ar=np.array(estimate_ar)
estimate_ar=estimate_ar.reshape(estimate_ar.shape[0],1)
print("estimator array shape")
print(estimate_ar.shape)
TestData=np.append(TestData,estimate_ar,axis=1)


dftesting=pd.DataFrame(TestData,columns=['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'])

df = pd.concat([dftraining, dftesting], join='outer', ignore_index=True)
df.to_csv("3challenge-1.csv")

