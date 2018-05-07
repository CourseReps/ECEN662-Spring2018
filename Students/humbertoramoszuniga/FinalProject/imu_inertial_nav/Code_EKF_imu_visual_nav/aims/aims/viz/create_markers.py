from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np
from numpy.linalg import eig

from aims.attkins import Quat

def create_frustrum(frame_name,f_size,f_color):
    """
    Create a frustrum marker that will be attached to `frame_name`. The
    size and color of the marker is controlled by `f_size` and `f_color`,
    respectively.

    This function works by first defining the coordinates of two squares.
    The first "small" square is attached to the XY plane and is centered on
    the z-axis. The second "large" square is parallel to the XY plane,
    centered on the z-axis, and translated by some amount in the positive
    z-direction.

    With these points defined, pairs of points are selected which parameterize
    the lines necessary to visually define and connect the two squares.
    """

    # define a line list
    frustrum_marker = Marker()
    frustrum_marker.header.frame_id = frame_name
    frustrum_marker.id=3
    frustrum_marker.type = Marker.LINE_LIST
    frustrum_marker.scale.x = f_size/10.
    frustrum_marker.color.a = 0.95
    frustrum_marker.color.r = f_color[0]/255.
    frustrum_marker.color.g = f_color[1]/255.
    frustrum_marker.color.b = f_color[2]/255.
    frustrum_marker.points = []

    # z-coordinate of large square
    z1=f_size*2
    # 1/2 times the side length of the large square
    n1=f_size

    # z-coordinate of the small square
    z2=f_size
    # 1/2 times the side length of the small square
    n2=f_size/2.

    # create a list of points which define the squares
    # the elements of the lists define the 4 points of the square
    # if you are looking at the XY plane, the order of points is quadrant
    # 1,2,3,4 respectively
    large_square_lists = [ [n1,n1,z1], [-n1,n1,z1], [-n1,-n1,z1], [n1,-n1,z1] ]
    small_square_lists = [ [n2,n2,z2], [-n2,n2,z2], [-n2,-n2,z2], [n2,-n2,z2] ]

    # convert the list of lists to a list of Point message objects
    large_square_points = [Point(*pi) for pi in large_square_lists]
    small_square_points = [Point(*pi) for pi in small_square_lists]

    # we need to define pairs of points which parameterize the lines we
    # want to draw
    # so, for each square...
    for point_list in [large_square_points,small_square_points]:
        # each square has points p0, p1, p2, p3
        # to draw the squares, we need lines with coordinates (p0,p1), (p1,p2),
        # (p2,p3), (p3,p0)
        plot_indices = [0,1,2,3,0]
        for point_index in range(1,5):
            first_point = plot_indices[point_index-1]
            frustrum_marker.points.append(point_list[first_point])
            next_point = plot_indices[point_index]
            frustrum_marker.points.append(point_list[next_point])

    # finally, we need to draw lines between the vertices of each square
    for l_point, s_point in zip(large_square_points,small_square_points):
        frustrum_marker.points.append(l_point)
        frustrum_marker.points.append(s_point)

    return frustrum_marker

def covariance_ellipse_marker(covariance,body_pose,color=(33,77,100)):

    var, R = eig(covariance)
    attitude=Quat.fromDCM(R.T)

    axes=np.sqrt(var)*3

    ellipse = Marker()
    ellipse.header.frame_id = "world"
    ellipse.type=Marker.SPHERE
    ellipse.scale.x = axes[0]
    ellipse.scale.y = axes[1]
    ellipse.scale.z = axes[2]
    ellipse.pose.orientation.x = attitude.x
    ellipse.pose.orientation.y = attitude.y
    ellipse.pose.orientation.z = attitude.z
    ellipse.pose.orientation.w = attitude.w
    ellipse.pose.position.x = body_pose.position[0,0]
    ellipse.pose.position.y = body_pose.position[1,0]
    ellipse.pose.position.z = body_pose.position[2,0]
    ellipse.color.a = 0.95
    ellipse.color.r = color[0]/255.
    ellipse.color.g = color[1]/255.
    ellipse.color.b = color[2]/255.
    return ellipse


def create_marker_from_features(features,rgb=(199,237,33),
                                marker_type=Marker.POINTS,marker_size=0.05):
    """

    """

    # define features marker
    features_marker = Marker()
    features_marker.header.frame_id = "world"
    features_marker.type = marker_type
    features_marker.scale.x = marker_size
    features_marker.scale.y = marker_size
    features_marker.color.a = 1.
    features_marker.color.r = rgb[0]/255.
    features_marker.color.g = rgb[1]/255.
    features_marker.color.b = rgb[2]/255.

    for feature in features:
        pos = feature.p_f_w.flatten()
        feature_point = Point()
        feature_point.x = pos[0]
        feature_point.y = pos[1]
        feature_point.z = pos[2]
        features_marker.points.append(feature_point)

    return features_marker
