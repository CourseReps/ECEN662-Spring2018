import numpy as np
from numpy import sqrt,log,cos,sin,pi
import matplotlib.pyplot as plt

mu,sigma = 0,1 #this is refered for the average and the scale


u1 = np.random.rand(1000)
u2 = np.random.rand(1000)

z1 = sqrt(-2*log(u1))*cos(2*pi*u2)
z2 = sqrt(-2*log(u1))*sin(2*pi*u2)

a = np.average(z1)

plt.subplot(221)
plt.hist(u1)
plt.subplot(222)
plt.hist(u2)
plt.subplot(223)
plt.hist(z1)
plt.subplot(224)
plt.hist(z2)
plt.show()


