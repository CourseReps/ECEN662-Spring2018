
import numpy as np

m=np.zeros(3)

# using https://www.ngdc.noaa.gov/geomag-web/?model=wmm#igrfwmm
# altitude 293 ft

# north nT
m[0]=24087.3	

# east nT
m[1]=1321.6

# vertical nT
m[2]=41016.4

# convert to guass
m/=1e4

