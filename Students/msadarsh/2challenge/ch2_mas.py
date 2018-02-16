import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# import challenge_func as cfunc 
from scipy.spatial import distance
import ch2_func as c2f              # challenge 2 function
import ch2_core as c2core           # core function of challenge 2. Used to plot the grid


df = pd.DataFrame.from_csv("2challenge.csv")
df0 = df.loc[df['label'] == 1.0] 
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)

# 0 and 1 in original code interchanged in the following code ie: df1,df2
TrainingData0 = df1.as_matrix(columns=None)
TrainingData1 = df0.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7',])
#print(TrainingData0[0])
tdarray=np.array(TrainingData0[0,0:8])
print(tdarray)

# print("printing the array")
# val=0
# for i in tdarray:
#     val+=1
#     print(i)
#     if math.isnan(i) :
#         print("yes nan")

########## code 
### Data0 - 0  Data1 - 1 
# training ratio
tr_ra= 0.8

# data 0 index 
dat0_i=math.floor(TrainingData0.shape[0]*tr_ra)
dat1_i=math.floor(TrainingData1.shape[0]*tr_ra)

print(TrainingData0.shape)

tr_dat0=TrainingData0[0:dat0_i,0:8]
tr_dat1=TrainingData1[0:dat1_i,0:8]

# print("minimum ele")
# print(tdarray[0])
# print(tdarray[6])
# print("minimum value")
# print(max(tdarray[0],tdarray[6]))

# ar1=np.array([[ ,5,1],[1, ,3],[7,8,2]])
# print("minimum value")
# print(ar1.min(axis=0))

min_ar_y=[]
max_ar_y=[]

tr_dat0_min=tr_dat0.min(axis=0);
tr_dat1_min=tr_dat1.min(axis=0);


# tr_dat0_max=tr_dat0.max(axis=0);
# tr_dat1_max=tr_dat1.max(axis=0);

# loop to calculate minimum and maximum of each column

for i in range(8) :
    min_ar_y=min_ar_y+[min(c2f.my_ar_minval(tr_dat0[:,i]),c2f.my_ar_minval(tr_dat1[:,i]))] 
    max_ar_y=max_ar_y+[max(c2f.my_ar_maxval(tr_dat0[:,i]),c2f.my_ar_maxval(tr_dat1[:,i]))] 

# print(tr_dat0_min)
# print(tr_dat1_min)

# print("val my func")
# print(min_ar_y)
# print(max_ar_y)

acuval_y=[2,2,2,2,2,2,2,2]

# c2core.coref(acuval_y,tr_dat0,tr_dat1,min_ar_y,max_ar_y,size_range)

c2core.coref(acuval_y,tr_dat0,tr_dat1,min_ar_y,max_ar_y,8,TrainingData0,TrainingData1,dat0_i,dat1_i,TestData,df0,df1)



