# Quaternion Kinematics (Quatematics) Module

Quatematics is simple library designed to streamline the implementation of the quaternion kinematic equations.
In addition it includes methods to aid in the creation of quaternions (e.g. random quaternions, quaternions from a direction cosine matrix), the conversion of quaternions to other attitude parameterizations, and basic quaternion operations (e.g. quaternion multiplication).

This module is primarily useful in an aerospace or mechanics context.
It has been throughly tested and used in various academic research settings.
The current implementation is designed for clarity but not for speed.
Upgrades and extensions are welcome.

The docstring of each method includes (or will soon include) a reference to the relevant equations used.
Most operations and method names are defined according to the following paper:

> Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter for 3D attitude estimation." University of Minnesota, Dept. of Comp. Sci. & Eng., Tech. Rep 2 (2005): 2005.

## Installation

The only dependency is numpy. Install numpy using your favorite method (e.g. pip, apt-get).

Clone this repository to a location of your choice.
Add quatematics to the PYTHONPATH by running setup.sh.

For example:

```bash
git clone https://github.com/dwhit15/quatematics
cd quatematics
sh setup.sh
```

## Status

This library has been used extensively on Ubuntu 14.04 with Python 2.7.
Performance on other platforms is unknown.

Feel free to use or contribute to this library in any way. 

## Usage

### Quaternion Creation and Manipulation

```python

import numpy as np

# import the library
from quatematics import Quat

# create a quaternion with a scalar component of 1 and all other components
# with a value of 0
identity_quat = Quat([0, 0, 0, 1])
# or
identity_quat = Quat(np.array([0, 0, 0, 1]) )
# or
identity_quat = Quat([1, 0, 0, 0], order="wxyz")
# or
identity_quat = Quat.eye

# create a random quaternion
rand_quat = Quat.rand()

# create a quaternion which represents a 90 deg rotation about the x-axis
q_x = Quat.fromAngleAxis(np.pi/2.,"x")
# create a quaternion which represents a 90 deg rotation about the y-axis
q_y = Quat.fromAngleAxis(np.pi/2.,"y")
# create a quaternion which represents a 90 deg rotation about the z-axis
q_z = Quat.fromAngleAxis(np.pi/2.,"z")

# now multiply those together to make a quaternion which represents a
# 3-rotation
q_xyz = q_x * q_y * q_z

# convert this quaternion to a direction cosine matrix (DCM)
C_xyz = q_xyz.asDCM()
# ... or a rotation matrix
R_xyz = q_xyz.asRM()

# make a quaternion from that DCM
my_quat = Quat.fromDCM(C_xyz)

# get the individual components of this quaternion
x = my_quat.x
y = my_quat.y
z = my_quat.z
w = my_quat.w

# get the imaginary components
img = my_quat.imaginary

# invert the quaternion
q_inv = my_quat.inverse()

# normalize the quaternion
my_quat.normalize()

# get the norm of the quaternion
q_norm = my_quat.norm()

# represent the quaternion as a column vector
q_vec = my_quat.asColVector()

```

### Some Kinematics

```python

import numpy as np

# import the library
from quatematics import Quat, AngularRate

# the AngularRate class is an inherited numpy array
# it is a 3x1 vector
# it is the same in every way to a standard numpy array
# except that it has 2 additional methods

# create an angular velocity vector
w = AngularRate([1,2,3])

# you can make AngularRate objects using the normal numpy methods
# but you need to "cast" them as AngularRate objects
# for example
w_zeros = np.zeros((3,1)).view(AngularRate)
w_ones = np.ones((3,1)).view(AngularRate)

# but once you have created an AngularRate you can use it just like a
# numpy array because it is a numpy array
# ...for example
add = w + w*3
mult = np.dot(w,w.T)

# make a quaternion
q = Quat.rand()

# the time rate of change of the quaternion is...
# see Trawny eq. 106
q_dot = np.dot(q.Xi(),w)

# ...it's also
# see Trawny eq. 107
q_dot = np.dot(w.Omega(),q.asColVector())

```
