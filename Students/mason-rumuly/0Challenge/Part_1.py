# Mason Rumuly
# Challenge 0
# Part 1 Main
#
# Generate sequence of independent standard Gaussian random variables using Box-Muller

import sys
import random
import numpy as np


# Returns two independent standard gaussian random variables
def box_muller():
    u0 = random.random()
    u1 = random.random()

    z0 = np.sqrt(-2 * np.log(u0)) * np.cos(2 * np.pi * u1)
    z1 = np.sqrt(-2 * np.log(u0)) * np.sin(2 * np.pi * u1)

    return [z0, z1]


# Take number of random variables to return
num = 2  # default to the first two
if len(sys.argv) > 1:
    num = int(sys.argv[1])

# Generate and display sequence of random variables
while num > 0:
    temp = box_muller()
    print(temp[0])
    num -= 1
    if num > 0:
        print(temp[1])
        num -= 1
