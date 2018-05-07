import numpy as np

from numpy.random import uniform, normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from aims import *


def uniform_spread(features,side_lengths):
    features[:] = uniform(side_lengths[0],side_lengths[1],size=np.shape(features))

def fill(features,kval):
    features.fill(kval)

def space_lab(num_features,lengths=(7,7,4)):

    all_features = XYZArray(num_features)

    x_length = lengths[0]
    y_length = lengths[1]
    z_length = lengths[2]

    x_coords = (-x_length,x_length)
    y_coords = (-y_length,y_length)
    z_coords = (0,z_length)

    depth = 0.5

    # wall on negative y axis
    minus_y = all_features[:,:num_features/3]
    uniform_spread(minus_y.x,x_coords)
    uniform_spread(minus_y.y,(-y_length-depth,-y_length+depth))
    uniform_spread(minus_y.z,z_coords)

    # features in front of wall
    other = all_features[:,num_features/3:]
    uniform_spread(other.x,x_coords)
    uniform_spread(other.y,(-2.5,-y_length))
    uniform_spread(other.z,(0,1))

    return all_features


def room_features(features_per_surface,lengths=(8,8,3)):
    all_features = XYZArray(features_per_surface*4)

    x_length = lengths[0]
    y_length = lengths[1]
    z_length = lengths[2]

    x_coords = (-x_length,x_length)
    y_coords = (-y_length,y_length)
    z_coords = (0,z_length)

    depth = 1

    # wall on positive y axis
    plus_y = all_features[:,0:features_per_surface]
    uniform_spread(plus_y.x,x_coords)
    uniform_spread(plus_y.y,(y_length-3,y_length+3))
    uniform_spread(plus_y.z,z_coords)

    # wall on positive x axis
    plus_x = all_features[:,features_per_surface:2*features_per_surface]
    uniform_spread(plus_x.x,(x_length-depth,x_length+depth))
    uniform_spread(plus_x.y,y_coords)
    uniform_spread(plus_x.z,z_coords)

    # wall on negative y axis
    minus_y = all_features[:,2*features_per_surface:3*features_per_surface]
    uniform_spread(minus_y.x,x_coords)
    uniform_spread(minus_y.y,(-y_length-depth,-y_length+depth))
    uniform_spread(minus_y.z,z_coords)

    # wall on negative x axis
    minus_x = all_features[:,3*features_per_surface:4*features_per_surface]
    uniform_spread(minus_x.x,(-x_length-depth,-x_length+depth))
    uniform_spread(minus_x.y,y_coords)
    uniform_spread(minus_x.z,z_coords)

    return all_features
