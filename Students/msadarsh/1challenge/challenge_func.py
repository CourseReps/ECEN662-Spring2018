import numpy as np


def gridlocate(pxval,pyval,gridx,gridy):
    in_grid=0
   #if ((((pxval>=gridx[0]) and (pxval<=gridx[-1])) and (pyval>=gridy[0])) and (pyval<=gridy[-1])) :   
    if (pxval>=gridx[0] and pxval<=gridx[-1]) and (pyval>=gridy[0] and pyval<=gridy[-1]) :
      in_grid=1
    else:
      in_grid=0  
    return in_grid  


def gridplot(grid_count,sq_valx,sq_valy,points_array):
# function to partition the grid and to count the number of points in each parition 
        
    for point_ind in range(points_array.shape[0]):
      p_xv=points_array[point_ind][0]
      p_yv=points_array[point_ind][1] 
      # checking for the x value in the grid 
      for i in range(sq_valx.size-1):
         if((p_xv>=sq_valx[i] and p_xv<sq_valx[i+1])or (i==sq_valx.size-2 and p_xv==sq_valx[i+1])):
           # checking for the y value in the grid
           for j in range(sq_valy.size-1):
              if((p_yv>=sq_valy[j] and p_yv<sq_valy[j+1]) or (j==sq_valy.size-2 and p_yv==sq_valy[j+1])):
                #  print("point at sq_valx=%f sq_valy=%f" % (sq_valx[i],sq_valy[j])) 
                 grid_count[i][j]+=1
                 break

           break

def grid_print(grid_count,sq_valx,sq_valy) :
    # print(grid_count)

    # print("00 01 10 11 -10")
    # print(points0_array[0][0])
    # print(points0_array[0][1])
    # print(points0_array[1][0])
    # print(points0_array[1][1])   

    # print(points0_array.T)
    # val_a=points0_array.T
    # #val_b=np.zeros((2,1))
    # val_b=[5,7]
    # val_b=np.array(val_b)
    # val_b=val_b.reshape(val_b.shape[0],1)
    # print(val_b.shape)
    # print("\n appended array\n")
    # print(np.append(val_a,val_b,axis=1))
    # # print(val_c)
    # print(val_b)

    # print(sq_valx.size)
    # m_sq_valx=np.array(sq_valx)
    # m_sq_valx=m_sq_valx.reshape(sq_valx.size,1)
    # print(m_sq_valx.shape)
    # print(m_sq_valx[1]+1)

    # ### appending to grid array
    gc=np.array(sq_valx)
    gc=gc.reshape(gc.shape[0],1)
    grid_count= np.append(gc[:-1],grid_count,axis=1)
    # print(grid_count)
    # print(grid_count.shape)
    # print(gc[:-1].shape)

    gr= np.array(sq_valy)
    gr=gr.reshape(gr.shape[0],1)
    minus_1=np.array([[-1]])
    # print(minus_1.shape)
    # print(gr.shape) 
    gr=np.append(minus_1,gr.T,axis=1)
    gr=gr.T 

    grid_count= np.append(gr[:-1],grid_count.T,axis=1)
    grid_count=grid_count.T
    # print(gr)
    # print(gr[:-1].shape)
    print(grid_count)


def grid_des_fill(grid_count0,grid_count1,grid_des,pr0,pr1):    
   #grid_count0 is for data =1 grid_count1
   for i in range(grid_count0.shape[0]):
      for j in range(grid_count0.shape[1]):
         if(grid_count0[i][j]*pr0 >= grid_count1[i][j]*pr1) :
            grid_des[i][j]=0
         else :
            grid_des[i][j]=1   

# end of grid_des_fill


# Generation of testresult

def gen_tst_res(grid_des,sq_valx,sq_valy,testdat,testres):

   for point_ind in range(testdat.shape[0]):
      p_xv=testdat[point_ind][0]
      p_yv=testdat[point_ind][1] 
      # checking for the x value in the grid 
      for i in range(sq_valx.size-1):
         if((p_xv>=sq_valx[i] and p_xv<sq_valx[i+1])or (i==sq_valx.size-2 and p_xv==sq_valx[i+1])):
           # checking for the y value in the grid
           for j in range(sq_valy.size-1):
              if((p_yv>=sq_valy[j] and p_yv<sq_valy[j+1]) or (j==sq_valy.size-2 and p_yv==sq_valy[j+1])):
                #  print("point at sq_valx=%f sq_valy=%f" % (sq_valx[i],sq_valy[j])) 
                 testres[point_ind]=grid_des[i][j]
                 break

           break


def point_res(grid_des,sq_valx,sq_valy,testdatx,testdaty):
      
      p_xv=testdatx
      p_yv=testdaty 
      # checking for the x value in the grid 
      for i in range(sq_valx.size-1):
         if((p_xv>=sq_valx[i] and p_xv<sq_valx[i+1])or (i==sq_valx.size-2 and p_xv==sq_valx[i+1])):
           # checking for the y value in the grid
           for j in range(sq_valy.size-1):
              if((p_yv>=sq_valy[j] and p_yv<sq_valy[j+1]) or (j==sq_valy.size-2 and p_yv==sq_valy[j+1])):
                #  print("point at sq_valx=%f sq_valy=%f" % (sq_valx[i],sq_valy[j])) 
                 return grid_des[i][j]
                 break

           break