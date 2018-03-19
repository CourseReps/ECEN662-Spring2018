import pandas as pd
import numpy as np
import ch2_func as c2f
from scipy.spatial import distance
import math

def coref(acuval_y,tr_dat0,tr_dat1,min_ar_y,max_ar_y,size_range,TrainingData0,TrainingData1,dat0_i,dat1_i,TestData,df0,df1) :
    pr0=0.5
    pr1=0.5
    sqval_y=[]     # contains the partition of the y_i coordinates
    for i in range(size_range):
        sqval_y=sqval_y+[np.linspace(min_ar_y[i],max_ar_y[i],acuval_y[i]+1)]
        # print(sqval_y)
    # print("indicator1")     
    # print(sqval_y)
    # print("indicator2") 
    # print("sizeof sqval_y[7]= %d" % len(sqval_y[7]))    
    # print("indicator3")
    # print(sqval_y[7][0])
   
    # initialising the grid
         
    grid_count0=np.zeros((acuval_y[0],acuval_y[1],acuval_y[2],acuval_y[3],acuval_y[4],acuval_y[5],acuval_y[6],acuval_y[7]))
    grid_count1=np.zeros((acuval_y[0],acuval_y[1],acuval_y[2],acuval_y[3],acuval_y[4],acuval_y[5],acuval_y[6],acuval_y[7]))
    grid_des=np.zeros((acuval_y[0],acuval_y[1],acuval_y[2],acuval_y[3],acuval_y[4],acuval_y[5],acuval_y[6],acuval_y[7]))
    
    # initialising array to mark point
    tr_dat0_mark=np.zeros((tr_dat0.shape[0],9))
    tr_dat1_mark=np.zeros((tr_dat1.shape[0],9))

    tr_dat0_mark_tst=np.zeros((tr_dat0.shape[0],9))
    tr_dat1_mark_tst=np.zeros((tr_dat1.shape[0],9))

    #plotting the grid
    
    c2f.gridplot_c2(grid_count0,acuval_y,tr_dat0,tr_dat0_mark,sqval_y,8,0.4)
    print("\n grid_count0 sum = %d \n" % grid_count0.sum())
    c2f.gridplot_c2(grid_count1,acuval_y,tr_dat1,tr_dat1_mark,sqval_y,8,0.6)
    print("\n grid_count1 sum = %d \n" % grid_count1.sum())
    print("To grid des")
    c2f.grid_des_fill_cf2(grid_count0,grid_count1,grid_des,pr0,pr1)


    testdat0=TrainingData0[dat0_i:,0:8]
    testres0=np.zeros((testdat0.shape[0],1))
    testdat1=TrainingData1[dat1_i:,0:8] 
    testres1=np.zeros((testdat1.shape[0],1))
    tr_dat0_mark_tst=np.zeros((testdat0.shape[0],9))
    tr_dat1_mark_tst=np.zeros((testdat1.shape[0],9))

    print("gridplot test0")
    c2f.gridplot_c2_tst(grid_des,acuval_y,testdat0,tr_dat0_mark_tst,sqval_y,8,0.4,testres0)
    print("gridplot test1")
    c2f.gridplot_c2_tst(grid_des,acuval_y,testdat1,tr_dat1_mark_tst,sqval_y,8,0.6,testres1)

     # evaluating the probability of error
    zero_error=0
    one_error=0
    for i in range(testres0.shape[0]):
       if math.isnan(testres0[i]):
           print("error in testres0") 
       if testres0[i] == 1 :
            zero_error+=1

    for i in range(testres1.shape[0]): 
        if math.isnan(testres1[i]):
           print("error in testres1")    
        if testres1[i]==0 :
            one_error+=1

            # evaluating p_e (probability of error)
    p_e=(zero_error+one_error)/float(testdat0.shape[0]+testdat1.shape[0])
            #est_grid[count_v][2]=p_e
            #count_v+=1
    print("probability of error= %f" % p_e)

    #finding mean
    mean0_s=TrainingData0.sum(axis=0)
    mean0_s=mean0_s/float(TrainingData0.shape[0])

    mean1_s=TrainingData1.sum(axis=0)
    mean1_s=mean1_s/float(TrainingData0.shape[0])
    
    mean0_s[6]=0.4
    mean0_s[7]=0.4
    
    mean1_s[6]=0.6
    mean1_s[7]=0.6
    mean0_s=mean0_s[:-1]
    mean1_s=mean1_s[:-1]
  
    TestData_ans=[]

    #TestData = dftest.as_matrix(columns=['Y0', 'Y1'])
    for tdv in range(TestData.shape[0]) :
      p_yv=[] 
      for pos in range(8): 
          if(math.isnan(TestData[tdv][pos])==False):
             p_yv=p_yv+[TestData[tdv][pos]]     
          else :  
             p_yv=p_yv+[0.5]         
      if(c2f.gridlocate_c2(p_yv,sqval_y)==1) :
        #c2f.gridplot_c2_res(grid_des,acuval_y,point,sqval_y,grid_depth)
        tmp=c2f.gridplot_c2_res(grid_des,acuval_y,p_yv,sqval_y,8)
        if (tmp is None) :
            print("none value at point %d " % tdv)
            print(p_yv)
        TestData_ans=TestData_ans+[tmp]
        # point0dis=point_distance
        # if(point0d)    
      else :
        
         if(distance.euclidean(p_yv,mean0_s)<=distance.euclidean(p_yv,mean1_s)):
            TestData_ans=TestData_ans+[0]
         else :
            TestData_ans=TestData_ans+[1] 
    print("printing testdata ans sum before conversion")
    print(sum(TestData_ans))
    # sumval=0
    # for ic in range(len(TestData_ans)):
    #     sumval=sumval+TestData_ans[i]

    # print("sumval=%d" % sumval)
    TestData_ans=np.array(TestData_ans)
    TestData_ans=TestData_ans.reshape(TestData_ans.shape[0],1)
    print("locator shape")
    print(TestData_ans.shape)
    TestData=np.append(TestData,TestData_ans,axis=1)
    # print("Test data row1")
    print(TestData[0:5])
    # print(distance.euclidean(mean0_s,mean1_s))
    dfdata=pd.DataFrame(TestData,columns=['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'])

    df = pd.concat([df0, df1, dfdata], join='outer', ignore_index=True)
    df.to_csv("2challenge.csv")