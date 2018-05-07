bag_directory = "/data/aims"
bag_name = "range_longer.bag"
range_topic_str = "/mavros/px4flow/ground_distance"


import aims
import numpy as np
from aims.attkins import Quat

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

tf=120.

t0=aims.tools.get_initial_time(bag_directory,bag_name,range_topic_str)
tf+=t0
save_time=(t0,tf)

# get range data
rangefinder = aims.tools.rangefinder_from_bag(bag_directory,bag_name,range_topic_str,save_time)
time,range_array = rangefinder.to_array()
range_array = range_array.flatten()

for i,range_element in enumerate(range_array):
    if abs(range_element) <= 1e-6:
        print "yo"
        range_array = np.delete(range_array,i)

hist,bins=np.histogram(range_array.flatten(),density=True)

plt.close("all")
plt.ion()

try:
    plt.style.use("dwplot")
except:
    print("Cannot use this stylesheet")

mu=np.mean(range_array)
var=np.var(range_array)
sigma=np.sqrt(var)
range_array-=mu

bins = np.linspace(-0.01, 0.01, 100)

# the histogram of the data
weights = np.ones_like(range_array)/float(len(range_array))
n, bins, patches = plt.hist(range_array, weights=weights,bins=bins, facecolor='green', alpha=0.75)
y = mlab.normpdf( bins, 0., sigma)
l = plt.plot(bins, y/range_array.shape[0], 'r--', linewidth=1)
plt.xlabel('Measurement')
plt.ylabel('Probability')
plt.title(r'Hmm')
#plt.axis([0.98,1.01,0,1])
plt.grid(True)
plt.show()