import numpy as np
import rospy
import tf
from visualization_msgs.msg import Marker, MarkerArray
from rosgraph_msgs.msg import Clock

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import time
from create_markers import *

from aims import Pose
from aims.attkins import Quat

class ROSViz(object):

    def __init__(self,estimator,clock,reference=None):

        # ekf instance that contains the data we want to display
        self.estimator = estimator
        # for simulated cases, display true feature positions
        self.true_feature_positions = None

        # estimated mapped feature positions
        self.est_feat_current_pub = rospy.Publisher("features/est/current",Marker,queue_size=10)
        # estimated feature positions that are no longer in map
        self.est_feat_past_pub = rospy.Publisher("features/est/past",Marker,queue_size=10)
        # esimated visible feature positions
        self.visible_features_pub = rospy.Publisher("features/est/visible",Marker,queue_size=10)
        self.features_ellipse_pub = rospy.Publisher("features/est/ellipse",MarkerArray,queue_size=10)
        # tf broadcaster
        self.tf_broadcast = tf.TransformBroadcaster()
        # publish 3 sigma ellipse
        self.cam_pose_ellipse_pub = rospy.Publisher("camera/ellipse",Marker,queue_size=10)
        # publish camera frustrum
        self.frustrum_pub = rospy.Publisher("camera/frustrum",Marker,queue_size=10)
        # publish simulated clock so that RVIZ time is synced with sim
        self.clock_pub = rospy.Publisher("/clock",Clock,queue_size=10)
        # reference clock
        self.aims_clock = clock
        # wall clock time
        # used to publish at max 30 hz according to wall clock
        self.last_publish_time = time.clock()

        # create frustrum marker
        self.frustrum = create_frustrum("camera/est",0.1,(255,255,255))

        # cv bridge
        self.cv_bridge = CvBridge()
        # publish debug image
        self.debug_img = rospy.Publisher("debug",Image,queue_size=10)

    def update(self):
        # ensure max 30 hz publish
        if (time.clock()-self.last_publish_time) >= 1/30.:

            # publish current camera pose
            camera_pose = self.estimator.camera_pose()
            self.tf_broadcast.sendTransform(
                camera_pose.position.flatten(),
                camera_pose.attitude.asColVector().flatten(),
                rospy.Time.now(),
                "camera/est",
                "world")

            # publish current imu pose
            self.tf_broadcast.sendTransform(
                self.estimator.state.p_i_w.flatten(),
                self.estimator.state.q_wi.asColVector().flatten(),
                rospy.Time.now(),
                "imu/est",
                "world")

            # keyframe pose
            if self.estimator.manager.kf_state_on:
                self.tf_broadcast.sendTransform(
                    self.estimator.state.kf_p_i_w.flatten(),
                    self.estimator.state.kf_q_wi.asColVector().flatten(),
                    rospy.Time.now(),
                    "keyframe/est",
                    "world")

            # find covariance of IMU pose
            slices = self.estimator.manager.slices
            P = self.estimator.P[slices.piw,slices.piw]
            # create 3 sigma ellipse centered on camera frame
            ellipse = covariance_ellipse_marker(P,camera_pose,(213, 234, 77))
            self.cam_pose_ellipse_pub.publish(ellipse)

            Pff = self.estimator.feature_covariance()

            # feature_ellipse_array = MarkerArray()
            # for i in xrange(self.estimator.manager.num_features):
            #     f_slice = slice(3*i,3*(1+i))
            #     Pfifi = Pff[f_slice,f_slice]
            #     p_fi = self.estimator.state.mapped_features[i].p_f_w
            #     pose_f = Pose()
            #     pose_f.position = p_fi
            #     pose_f.attitude = Quat.eye
            #     feature_ellipse_array.markers.append(covariance_ellipse_marker(Pfifi,pose_f,(213, 234, 77)))
            # self.features_ellipse_pub.publish(feature_ellipse_array)

            # update ros time
            clock_msg = Clock()
            clock_msg.clock = rospy.Time(self.aims_clock.now())
            self.clock_pub.publish(clock_msg)

            # make lits that distinguish between features that are...
            visible_feature_ids = [feature.id for feature in self.estimator.visible_features]
            # ...in the map but were not just re-observed
            mapped_features = [feature for feature in self.estimator.state.mapped_features if feature.id not in visible_feature_ids]
            # in the map and just re-observed
            visible_features = [feature for feature in self.estimator.state.mapped_features if feature.id in visible_feature_ids]
            # features that were mapped but have now been "forgotten"
            old_features = [feature for feature in self.estimator.state.old_features]

            # display visible features
            if len(visible_features) > 0:
                self.visible_features_pub.publish(
                    create_marker_from_features(visible_features,(0, 255, 0)))

            # display mapped features
            if len(mapped_features) > 0:
                self.est_feat_current_pub.publish(
                    create_marker_from_features(mapped_features,(255, 255, 0)))

            # publish old features
            if len(old_features) > 0:
                self.est_feat_past_pub.publish(
                    create_marker_from_features(old_features,(255, 0, 0)))

            # publish frustrum
            self.frustrum_pub.publish(self.frustrum)

            # publish debug image
            img = self.estimator.image_feature_tracker.output_image
            if isinstance(img,np.ndarray):
                self.debug_img.publish(
                    self.cv_bridge.cv2_to_imgmsg(img, "bgr8"))
            img = self.estimator.sim_feature_tracker.output_image
            if isinstance(img,np.ndarray):
                self.debug_img.publish(
                    self.cv_bridge.cv2_to_imgmsg(img, "bgr8"))

            # update wall time
            self.last_publish_time = time.clock()
