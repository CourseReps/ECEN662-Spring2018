# Mason Rumuly
# Challenge 0
# Part 2 Main
#
# Generate sequence of independent standard Gaussian random variables using Marsaglia

import sys
import random
import numpy as np


# Returns two independent standard gaussian random variables
def marsaglia():
    while True:
        u0 = random.random()
        u1 = random.random()
        s = pow(u0, 2) + pow(u1, 2)
        if s < 1:
            t = np.sqrt(-2 * np.log(s) / s)
            return [u0 * t, u1 * t]


# Take number of random variables to return
num = 2  # default to the first two
if len(sys.argv) > 1:
    num = int(sys.argv[1])

# Generate and display sequence of random variables
while num > 0:
    temp = marsaglia()
    print(temp[0])
    num -= 1
    if num > 0:
        print(temp[1])
        num -= 1

