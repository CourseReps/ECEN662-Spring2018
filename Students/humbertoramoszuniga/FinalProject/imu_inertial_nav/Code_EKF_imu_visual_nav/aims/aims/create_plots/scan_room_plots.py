import os
from aims import *
import matplotlib.pylab as plt
from multiplot2d import MultiPlotter
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from aims.attkins import Quat

mpl.rcParams['legend.fontsize'] = 8

default = database_from_file(os.environ['HOME']+"/Dropbox/aims_db/" + "scan_room_final_2" + ".odb")
up_down = database_from_file(os.environ['HOME']+"/Dropbox/aims_db/" + "scan_room_up_down_final_2" + ".odb")

fig = plt.figure(figsize=(6,3))
ax = fig.gca(projection='3d')

l=4000
m=12800
X = up_down.idb.reference.kinematics.position.x[l:m]
Y = up_down.idb.reference.kinematics.position.y[l:m]
Z = up_down.idb.reference.kinematics.position.z[l:m]
ax.plot(X,
       Y , 
Z, label='With Guidance')
ax.plot(default.idb.reference.kinematics.position.x[l:m], 
        default.idb.reference.kinematics.position.y[l:m], 
    default.idb.reference.kinematics.position.z[l:m], label='No Guidance')


ax.set_zlim(0,3)
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)

num = 35
quiver_k=np.linspace(l,m,num)

quiver_pts = XYZArray(num)
quiver_dir = XYZArray(num)

for i, k in enumerate(quiver_k):
    quiver_pts[:,i] = up_down.idb.reference.kinematics.position[:,k]
    quiver_dir[:,i] = Quat(up_down.idb.reference.kinematics.attitude[:,k]).asRM()[:,0]

ax.quiver(quiver_pts.x,quiver_pts.y,quiver_pts.z,quiver_dir.x,quiver_dir.y,quiver_dir.z,pivot="tail",arrow_length_ratio=.5,length=0.5,color="k",linewidth=1.,label="Camera Pointing Direction")
ax.legend(ncol=3,loc="upper center")
ax.locator_params( nbins=6)
fig.tight_layout()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_aspect(0.8)
# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
## Comment or uncomment following both lines to test the fake bounding box:
#for xb, yb, zb in zip(Xb, Yb, Zb):
#   ax.plot([xb], [yb], [zb], 'w')

save_dir = os.environ['HOME']+"/Dropbox/whitten_thesis/tex/fig/" + "scan_room_trajectory.pdf" 
#fig.savefig(save_dir, bbox_inches='tight',pad_inches=1.,dpi=300,format="pdf")
fig.savefig(save_dir, transparent=True,dpi=300,format="pdf")



plt.ion()
plt.show()

