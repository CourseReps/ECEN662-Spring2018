import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import challenge_func as cfunc 
from scipy.spatial import distance

df = pd.DataFrame.from_csv("1challenge.csv")
df0 = df.loc[df['label'] == 1.0]
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)


TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

print(TrainingData0.shape)
print(TestData.shape)
# plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
# plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
# plt.plot(TestData[:,0], TestData[:,1], 'o', color='k')
# plt.axis('equal')
# plt.show()

########## code 
### Data0 - 1  Data1 - 0 
# training ratio
tr_ra= 0.8

# data 0 index 
dat0_i=math.floor(TrainingData0.shape[0]*tr_ra)
dat1_i=math.floor(TrainingData1.shape[0]*tr_ra)
#print("dat0_i= "+ str(dat0_i))
#print("dat1_i= "+ str(dat1_i))

tr_dat0=TrainingData0[0:dat0_i,0:2]
tr_dat1=TrainingData1[0:dat1_i,0:2]

minx_data=min(tr_dat0.min(axis=0)[0],tr_dat1.min(axis=0)[0])
maxx_data=max(tr_dat0.max(axis=0)[0],tr_dat1.max(axis=0)[0])

miny_data=min(tr_dat0.min(axis=0)[1],tr_dat1.min(axis=0)[1])
maxy_data=max(tr_dat0.max(axis=0)[1],tr_dat1.max(axis=0)[1])

p0_vals=np.arange(0.1,1,0.1)
gs_vals=np.arange(10,100,10)
ex_sz=len(p0_vals)*len(gs_vals)

est_grid=np.zeros((ex_sz,3))
count_v=0
#print([minx_data,maxx_data,miny_data,maxy_data])
for gs in range(1) :
   
    for p0 in range(1) :
            #est_grid[count_v][0]=gs_vals[gs]
            #est_grid[count_v][1]=p0_vals[p0] 
            # accuracy index
            x_acuval=10
            y_acuval=10



            sq_valx=np.linspace(minx_data,maxx_data,x_acuval+1)

            sq_valy=np.linspace(miny_data,maxy_data,y_acuval+1)

            #grid for 1
            grid_count0=np.zeros((x_acuval,y_acuval))
            #grid for 0
            grid_count1=np.zeros((x_acuval,y_acuval))
            # grid for decision
            grid_des=np.zeros((x_acuval,y_acuval))

            #filling the grid
            cfunc.gridplot(grid_count0,sq_valx,sq_valy,tr_dat1)
            #print("grid_count0       ="+str(grid_count0.sum()))
            # grid_count0 - probability distribution for 0
            grid_count0=grid_count0/grid_count0.sum()

            # grid_count1 - probability distribution for 1
            cfunc.gridplot(grid_count1,sq_valx,sq_valy,tr_dat0)
            #print("grid_count1 ="+str(grid_count1.sum()))
            grid_count1=grid_count1/grid_count1.sum()

            #print(grid_count0.sum())

            # Probability distribution estimate
            
            pr0=0.6
            pr1=0.4

            # filling grid_des
            cfunc.grid_des_fill(grid_count0,grid_count1,grid_des,pr0,pr1)
            #print(grid_des)

            # using decision rule on data. testdat0 for 0 . testdat1 for 1.

            testdat0=TrainingData1[dat1_i:,0:2]
            testres0=np.zeros((testdat0.shape[0],1))
            testdat1=TrainingData0[dat0_i:,0:2] 
            testres1=np.zeros((testdat1.shape[0],1))


            #using test data for 0 ,1 and generating test result
            #cfunc.gen_tst_res(grid_des,sq_valx,sq_valy,testdat0,testres0,testsz)
            cfunc.gen_tst_res(grid_des,sq_valx,sq_valy,testdat0,testres0)
            cfunc.gen_tst_res(grid_des,sq_valx,sq_valy,testdat1,testres1)

            #print("testres0 sz= %f" % testres0.shape[0])
            #print("testres1 sz= %f" % testres1.shape[0])
            # evaluating the probability of error
            zero_error=0
            one_error=0
            for i in range(testres0.shape[0]):
              if testres0[i] == 1 :
                zero_error+=1

            for i in range(testres1.shape[0]):    
                if testres1[i]==0 :
                  one_error+=1

            # evaluating p_e (probability of error)
            p_e=(zero_error+one_error)/float(testdat0.shape[0]+testdat1.shape[0])
            #est_grid[count_v][2]=p_e
            #count_v+=1
            print("probability of error= %f" % p_e)
            
            #loop this if necessary      


#finding the best estimate

# print("min val combination")
# est_pos=np.argmin(est_grid,axis=0)
# print(est_grid[est_pos[2]])

# print(est_grid)  

#working on test data

#finding mean
mean0_s=TrainingData1.sum(axis=0)
mean0_s=mean0_s/float(TrainingData1.shape[0])
mean0_x=mean0_s[0]
mean0_y=mean0_s[1]

mean1_s=TrainingData0.sum(axis=0)
mean1_s=mean1_s/float(TrainingData0.shape[0])
mean1_x=mean1_s[0]
mean1_y=mean1_s[1]

print(mean0_s)
print(mean1_s)

TestData_ans=[]

#TestData = dftest.as_matrix(columns=['Y0', 'Y1'])
for tdv in range(TestData.shape[0]) :          
    if(cfunc.gridlocate(TestData[tdv][0],TestData[tdv][1],sq_valx,sq_valy)==1) :
        tmp=cfunc.point_res(grid_des,sq_valx,sq_valy,TestData[tdv][0],TestData[tdv][1])
        TestData_ans=TestData_ans+[tmp]
        # point0dis=point_distance
        # if(point0d)    
    else :
         if(distance.euclidean(TestData[0],[mean0_x,mean0_y])<=distance.euclidean(TestData[0],[mean1_x,mean1_y])):
            TestData_ans=TestData_ans+[0]
         else :
             TestData_ans=TestData_ans+[1] 


# 
# print(TestData_ans)
# print(len(TestData_ans))    
# print(TestData)
TestData_ans=np.array(TestData_ans)
TestData_ans=TestData_ans.reshape(TestData_ans.shape[0],1)
# print(TestData_ans.shape)
TestData=np.append(TestData,TestData_ans,axis=1)
# print("Test data row1")
print(TestData[0:5])

dfdata=pd.DataFrame(TestData,columns=['Y0', 'Y1','label'])

df = pd.concat([df0, df1, dfdata], join='outer', ignore_index=True)
df.to_csv("1challenge.csv")
