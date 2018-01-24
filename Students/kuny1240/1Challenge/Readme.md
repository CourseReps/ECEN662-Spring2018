## Challenge 1 for Kun Yang
___
In this challenge I have learned things about both python and binary detection.
### Python:
As my self write more of Matlab code before it is really hard for me to use numpy at first, there are some important functions
in numpy to compute array.
___
```python
import numpy as np

np.linalg.det() # this is the function to compute determinants of a matrix
np.dot()  # this is the dot multiply
np.hstack() # this is the function to connect two different matrix with same rows
```
___
### Binary Detection:
I learned the importance to do some assumptions if your problem parameters are insufficient.
Here are my assumptions:
1. __Regard data frequency as the probablity of an attibute__
2. __Regard all two distributions as 2-dimentional Gaussian distribution__
___
I also tried to do some estimation of the parameters for the PDF of a 2-dimentional Gaussian distribution, using 
_maximum likelihood estimation_ which is a method I only understanded a part. I will try to understand it in the
following days.
