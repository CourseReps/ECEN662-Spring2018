# functions for ch2_mas
import math
import numpy as np

def gridlocate_c2(py_v,sqval_y):
    in_grid=0
   #if ((((pxval>=gridx[0]) and (pxval<=gridx[-1])) and (pyval>=gridy[0])) and (pyval<=gridy[-1])) :   
    if ((py_v[0]>=sqval_y[0][0] and py_v[0]<=sqval_y[0][-1]) and (py_v[1]>=sqval_y[1][0] and py_v[0]<=sqval_y[1][-1]) and (py_v[2]>=sqval_y[2][0] and py_v[0]<=sqval_y[2][-1]) and (py_v[3]>=sqval_y[3][0] and py_v[3]<=sqval_y[3][-1]) and (py_v[4]>=sqval_y[4][0] and py_v[4]<=sqval_y[4][-1]) and (py_v[5]>=sqval_y[5][0] and py_v[5]<=sqval_y[5][-1]) and (py_v[6]>=sqval_y[6][0] and py_v[6]<=sqval_y[6][-1]) and (py_v[7]>=sqval_y[7][0] and py_v[7]<=sqval_y[7][-1]) )  :
      in_grid=1
    else:
      in_grid=0  
    return in_grid  




# The following function is especially for input data. To find minimum of 
# each column
def my_ar_minval(array1):

    min=array1[0]
    for i in range(len(array1)):
        if (math.isnan(min)==True) or (array1[i]<min and (math.isnan(array1[i])==False))  :
            min=array1[i]
    return min


# The following function is especially for input data. To find maximum of 
# each column

def my_ar_maxval(array1):
    
    max=array1[0]
    for i in range(len(array1)):
        if (math.isnan(max)==True) or (array1[i]>max and (math.isnan(array1[i])==False))  :
            max=array1[i]
    return max    


def coordinate_plot(p_yv,cp_pos,tr_dat0_mark,point_ind,sq_valy,rep_val):
    if (math.isnan(p_yv[cp_pos])==True): 
        p_yv[cp_pos]=rep_val

    for j in range(len(sq_valy[cp_pos])-1):
        if((p_yv[cp_pos]>=sq_valy[cp_pos][j] and p_yv[cp_pos]<sq_valy[cp_pos][j+1]) or (j==len(sq_valy[cp_pos])-2 and p_yv[cp_pos]==sq_valy[cp_pos][j+1])): 
           tr_dat0_mark[point_ind][cp_pos+1]=j
           return 1
 

def gridplot_c2(grid_count,acuval_y,points_array,tr_dat0_mark,sqval_y,grid_depth,rep_val):
    print("\nin gridplot_c2\n")
    c_tst_val=0
    for point_ind in range(points_array.shape[0]):
      p_yv=[] 
      for pos in range(grid_depth):
          p_yv=p_yv+[points_array[point_ind][pos]]

      for cp_pos in range(grid_depth):    
          cprv=coordinate_plot(p_yv,cp_pos,tr_dat0_mark,point_ind,sqval_y,rep_val)
          if (cprv!=1) :
               
               print("not equal")
               print(p_yv)
               print("point")
               tr_dat0_mark[point_ind][cp_pos+1]=0
               break
          if (cp_pos==grid_depth-1) :
              c_tst_val+=1
              grid_count[int(tr_dat0_mark[point_ind][1])][int(tr_dat0_mark[point_ind][2])][int(tr_dat0_mark[point_ind][3])][int(tr_dat0_mark[point_ind][4])][int(tr_dat0_mark[point_ind][5])][int(tr_dat0_mark[point_ind][6])][int(tr_dat0_mark[point_ind][7])][int(tr_dat0_mark[point_ind][8])]+=1                
              tr_dat0_mark[point_ind][0]=1
            #   print(tr_dat0_mark[point_ind])

    print("c_tst_val = %d" % c_tst_val)          


def grid_des_fill_cf2(grid_count0,grid_count1,grid_des,pr0,pr1):
   val_des=0     
   #grid_count0 is for data =1 grid_count1
   for i0 in range(grid_count0.shape[0]):
      for i1 in range(grid_count0.shape[1]):
          for i2 in range(grid_count0.shape[2]):
              for i3 in range(grid_count0.shape[3]):
                  for i4 in range(grid_count0.shape[4]):
                      for i5 in range(grid_count0.shape[5]):
                          for i6 in range(grid_count0.shape[6]):
                              for i7 in range(grid_count0.shape[7]):
                                   if(grid_count1[i0][i1][i2][i3][i4][i5][i6][i7]*pr1 >= grid_count0[i0][i1][i2][i3][i4][i5][i6][i7]*pr0) :
                                       grid_des[i0][i1][i2][i3][i4][i5][i6][i7]=1
                                    #    val_des+=1
                                   else :
                                       grid_des[i0][i1][i2][i3][i4][i5][i6][i7]=0   
                                    #    val_des+=1
#    print("valdes = %d" % val_des)



################

def coordinate_plot_tst(p_yv,cp_pos,tr_dat0_mark,point_ind,sq_valy,rep_val):
    if (math.isnan(p_yv[cp_pos])==True): 
        p_yv[cp_pos]=rep_val

    for j in range(len(sq_valy[cp_pos])-1):
        if((p_yv[cp_pos]>=sq_valy[cp_pos][j] and p_yv[cp_pos]<sq_valy[cp_pos][j+1]) or (j==len(sq_valy[cp_pos])-2 and p_yv[cp_pos]==sq_valy[cp_pos][j+1])): 
           tr_dat0_mark[point_ind][cp_pos+1]=j
           return 1
 

def gridplot_c2_tst(grid_des,acuval_y,points_array,tr_dat0_mark,sqval_y,grid_depth,rep_val,testres):
    # print("\nin gridplot_c2\n")
    c_tst_val=0
    for point_ind in range(points_array.shape[0]):
      p_yv=[] 
      for pos in range(grid_depth):
          p_yv=p_yv+[points_array[point_ind][pos]]

      for cp_pos in range(grid_depth):    
          cprv=coordinate_plot_tst(p_yv,cp_pos,tr_dat0_mark,point_ind,sqval_y,rep_val)
          if (cprv!=1) :
               print("before not equal")
               print(cp_pos)
               print("not equal")
               print(p_yv)
               print("point")
               tr_dat0_mark[point_ind][cp_pos+1]=0
          if (cp_pos==grid_depth-1) :
              c_tst_val+=1
              testres[point_ind]=grid_des[int(tr_dat0_mark[point_ind][1])][int(tr_dat0_mark[point_ind][2])][int(tr_dat0_mark[point_ind][3])][int(tr_dat0_mark[point_ind][4])][int(tr_dat0_mark[point_ind][5])][int(tr_dat0_mark[point_ind][6])][int(tr_dat0_mark[point_ind][7])][int(tr_dat0_mark[point_ind][8])]                
              tr_dat0_mark[point_ind][0]=1
            #   print(tr_dat0_mark[point_ind])

######## fuction to get value for a test point    

def coordinate_plot_res(p_yv,cp_pos,tr_mark,sq_valy):
    if (math.isnan(p_yv[cp_pos])==True): 
        p_yv[cp_pos]=0.5

    for j in range(len(sq_valy[cp_pos])-1):
        if((p_yv[cp_pos]>=sq_valy[cp_pos][j] and p_yv[cp_pos]<sq_valy[cp_pos][j+1]) or (j==len(sq_valy[cp_pos])-2 and p_yv[cp_pos]==sq_valy[cp_pos][j+1])): 
           #tr_dat0_mark[point_ind][cp_pos+1]=j
        #    print("adding tr_mark %d" % j )
           tr_mark[0][cp_pos]=j
           return 1
 

def gridplot_c2_res(grid_des,acuval_y,point,sqval_y,grid_depth):
    # print("\nin gridplot_c2_res\n")
    tr_mark=np.zeros((1,grid_depth))
    c_tst_val=0
    p_yv=[] 
    for pos in range(grid_depth):
       p_yv=p_yv+[point[pos]]

    for cp_pos in range(grid_depth):    
        cprv=coordinate_plot_res(p_yv,cp_pos,tr_mark,sqval_y)
        if (cprv!=1) :
               print("before not equal")
               print(cp_pos)
               print("not equal")
               print(p_yv)
               print("point")
               tr_mark[0][cp_pos]=0
        if (cp_pos==grid_depth-1) :
              c_tst_val+=1
            #   print(tr_mark)
            #   print("grid_des=")
            #   print(grid_des[int(tr_mark[0][0])][int(tr_mark[0][1])][int(tr_mark[0][2])][int(tr_mark[0][3])][int(tr_mark[0][4])][int(tr_mark[0][5])][int(tr_mark[0][6])][int(tr_mark[0][7])])
              return (grid_des[int(tr_mark[0][0])][int(tr_mark[0][1])][int(tr_mark[0][2])][int(tr_mark[0][3])][int(tr_mark[0][4])][int(tr_mark[0][5])][int(tr_mark[0][6])][int(tr_mark[0][7])])                
            #   print(tr_dat0_mark[point_ind])

# print("c_tst_val = %d" % c_tst_val) 
