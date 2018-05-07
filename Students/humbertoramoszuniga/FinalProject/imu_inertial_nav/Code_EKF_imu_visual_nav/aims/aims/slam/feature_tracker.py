
import math

# from measurement import FeatureMeasurement, PixelFeature
from copy import deepcopy
import numpy as np
import cv2
from aims.core import Pose
from numpy.linalg import norm

from itertools import compress

def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1

class Feature(object):

    def __init__(self):
        self.descriptor = None
        self.pt = None

class Candidate(object):

    def __init__(self):
        self.train = Feature()
        self.query = Feature()
        self.id = None
        self.p_f_w = None


class FeatureTracker(object):

    def __init__(self):
        self.inlier_list = np.zeros(5,dtype=np.uint8)
        self.inlier_index = 0
        self.len_inlier_history = 0
        self.new_kf_desired = True
        self.assign_id = 0

    def update_inlier_history(self,inlier):
        inlier_buffer_size = self.inlier_list.shape[0]

        self.inlier_list[self.inlier_index] = inlier
        if self.inlier_index == inlier_buffer_size-1:
            self.inlier_index = 0
        else:
            self.inlier_index += 1

        if self.len_inlier_history < inlier_buffer_size:
            self.len_inlier_history += 1

    def reset_inlier_history(self):
        self.len_inlier_history = 0
        self.inlier_index = 0


    def declare_new_keyframe(self):
        if self.len_inlier_history < self.inlier_list.shape[0]:
            return False

        if np.mean(self.inlier_list) <= self.new_keyframe_threshold:
            return True
        else:
            return False

class SimulatedFeatureTracker(FeatureTracker):

    def __init__(self,image_size,new_keyframe_threshold):
        self.new_keyframe_threshold = new_keyframe_threshold
        self.image_size=image_size
        self.output_image = None
        super(SimulatedFeatureTracker,self).__init__()

        self.kf_features = dict()

    def register_features(self,feature_measurement, mapped_features):

        new_kf_declared = False
        candidates = []

        self.output_image = np.ones((self.image_size[1],self.image_size[0],3),
                                     dtype=np.uint8)*255
        cv2.line(self.output_image, (0,self.image_size[1]),
                 (self.image_size[0],self.image_size[1]), (0, 0, 0), 2)

        if self.new_kf_desired == True:
            self.kf_features = dict()
            for feature in feature_measurement.features:
                self.kf_features[feature.id] = np.array([feature.u,feature.v ])
                self.new_kf_desired = False
                self.reset_inlier_history()
                new_kf_declared = True

        mapped_ids = []
        for mapped in mapped_features:
            mapped_ids.append(mapped.id)
            mapped.measurement = None
        for feature in feature_measurement.features:
            if feature.id in mapped_ids:
                i = mapped_ids.index(feature.id)
                ym = np.array([[feature.u,feature.v ]]).T
                mapped_features[i].measurement = ym
                self.draw_mapped(ym)
            elif feature.id in self.kf_features.keys():
                candidate = Candidate()
                candidate.query.pt = self.kf_features[feature.id]
                candidate.train.pt = np.array([feature.u,feature.v ])
                candidate.train.var = np.array([0.04, 0.04])
                candidate.query.var = np.array([0.04, 0.04])
                candidate.id = feature.id
                candidates.append(candidate)
                self.draw_unmapped(candidate.query.pt,candidate.train.pt)


        self.update_inlier_history(len(candidates))
        if self.declare_new_keyframe():
            self.new_kf_desired = True

        return candidates,new_kf_declared

    def draw_mapped(self,pt):
        (x2,y2) = pt
        cv2.circle(self.output_image, (int(x2),int(y2)), 3, (0, 255, 0), -1)

    def draw_unmapped(self,p_kf,p_now):
        (x1,y1) = p_kf
        (x2,y2) = p_now
        cv2.line(self.output_image, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 2)


class ImageFeatureTracker(FeatureTracker):

    def __init__(self,new_keyframe_threshold):
        self.new_keyframe_threshold = new_keyframe_threshold
        self.output_image = None
        self.orb_params = dict(
            patchSize=30,
            edgeThreshold=30)

        super(ImageFeatureTracker,self).__init__()

    def define_new_img(self,input_image):
        """
        If new image is color, create black and white version for
        feature extraction.
        Save the color copy (or create one if necessary) for output.
        """

        # if image is grayscale, make an output image that will support color
        if input_image.ndim == 2:
            self.new_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            self.new_image = input_image
        # if image is color, convert it to black and white
        elif input_image.ndim == 3:
            self.new_image_rgb = input_image
            self.new_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # sharpness serves as criteria for various functions
        self.sharpness = cv2.Laplacian(self.new_image, cv2.CV_64F).var()

    def get_orb_n(self,num_features,img):
        """
        Get an orb object with the provided number of features. All other
        orb parameters are the default parameters of this class.
        """
        orb = cv2.ORB(nfeatures=num_features,**self.orb_params)

        mask = 255*np.ones_like(img,dtype=np.uint8)

        keypoints_1, descriptors_1 = orb.detectAndCompute(img,None)
        for kp in keypoints_1:
            (x2,y2) = kp.pt
            cv2.circle(mask, (int(x2),int(y2)), 15, (0, 0, 0), -1)

        orb = cv2.ORB(nfeatures=int(num_features*1.5),**self.orb_params)
        keypoints_2, descriptors_2 = orb.detectAndCompute(img, mask)

        keypoints = keypoints_1 + keypoints_2
        descriptors = np.vstack((descriptors_1,descriptors_2))

        return keypoints, descriptors


    def save_as_keyframe(self):
        """
        If criteria for new keyframe exist, create a new keyframe.
        Returns true if current image is used as keyframe.
        """
        if self.new_kf_desired:
            if self.sharpness >= 50.:
                self.kf_image = self.new_image
                self.kf_image_rgb = cv2.cvtColor(self.kf_image, cv2.COLOR_GRAY2BGR)

                self.new_kf_desired = False
                self.reset_inlier_history()
                for feature in self.mapped_features:
                    feature.keyframe += 1

                self.kf_keypoints, self.kf_descriptors = self.get_orb_n(500,self.kf_image)
                return True

        return False

    def ready_for_matching(self):
        """
        Determines whether or not the new image meets the criteria
        necessary for feature matching.
        """
        if self.sharpness >= 50:
            return True
        else:
            return False

    def create_output_image_base(self):
        """
        Stack the keyframe and latest image in color format.
        """
        self.output_image = np.vstack((self.kf_image_rgb,self.new_image_rgb))

    def filter_with_ratio_test(self,matches,ratio):
        """
        Returns a list of matches that pass the ratio test.
        """
        good=[]
        for match in matches:
            if len(match)==2:
                m,n=match
                if m.distance < ratio*n.distance:
                    good.append(m)
        return good

    def matches_to_pts(self,matches):
        kf_pts = np.float32([ self.kf_keypoints[m.queryIdx].pt for m in matches ]).reshape(1,-1,2)
        new_pts = np.float32([ self.new_keypoints[m.trainIdx].pt for m in matches ]).reshape(1,-1,2)
        return kf_pts,new_pts

    def find_new_matches(self):
        """
        Find high quality matches between keyframe and current image in
        regions that are "far" from currently mapped features.
        """

        candidates = []

        # get raw matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(self.kf_descriptors,self.new_descriptors,k=2)

        # filter out matches that don't pass ratio test
        matches = self.filter_with_ratio_test(matches,0.55)


        # if there are more than 10 matches, find the best matches by
        # filtering outliers from dataset using RANSAC
        if len(matches) > 10:
            # get pixel coordinates of matches in the keyframe and new image
            # result is a 1xnx2 matrix
            # opencv likes this odd format
            kf_pts,new_pts = self.matches_to_pts(matches)

            # find fundamental matrix
            F_mat, mask = cv2.findFundamentalMat(kf_pts, new_pts, cv2.FM_8POINT)

            kf_pts_ideal, new_pts_ideal = cv2.correctMatches(F_mat,kf_pts,new_pts)
            new_pts_error = new_pts-new_pts_ideal
            if not np.isnan(new_pts_error).any():
                kf_pts_error = kf_pts-kf_pts_ideal
                # variance of expected error in pixel measurement
                new_pts_var = np.var(new_pts_error[0],axis=0)
                kf_pts_var = np.var(kf_pts_error[0],axis=0)

                # sort the matches so that the best matches are at the front of
                # the list
                matches = sorted(matches, key = lambda x:x.distance)

                radius = 25.

                # best contains the top matches that are "far" from other features
                best = []
                for match in matches:
                    # for now assume that this match is good
                    add = True
                    p1 = np.array(self.new_keypoints[match.trainIdx].pt)
                    # check if this match is "far" from other matches
                    for already_best in best:
                        p2 = np.array(self.new_keypoints[already_best.trainIdx].pt)
                        if np.linalg.norm(p2-p1) <= radius:
                            add=False
                            break
                    # check if htis match is "far" from mapped features
                    # for feature in [feature for feature in self.mapped_features if feature.keyframe < 5]:
                    for feature in self.mapped_features:
                        p2 = feature.prediction.flatten()
                        if np.linalg.norm(p2-p1) <= radius:
                            add=False
                            break
                    if add:
                        best.append(match)

                # any matches that have made it this far will be considered
                # candidates
                for match in best:
                    candidate = Candidate()
                    candidate.query.descriptor = self.kf_descriptors[match.queryIdx]
                    candidate.query.pt = np.array(self.kf_keypoints[match.queryIdx].pt)
                    candidate.query.var = kf_pts_var
                    # print kf_pts_var
                    candidate.train.descriptor = self.new_descriptors[match.trainIdx]
                    candidate.train.pt = np.array(self.new_keypoints[match.trainIdx].pt)
                    candidate.train.var = new_pts_var
                    candidate.id = self.assign_id
                    self.assign_id += 1
                    candidates.append(candidate)

                self.draw_unmapped(best)

        return candidates


    def find_mapped(self):

        mapped_descriptors = np.zeros((len(self.mapped_features),32),dtype=np.uint8)

        num_mapped = len(self.mapped_features)
        if num_mapped > 0:
            for i,feature in enumerate(self.mapped_features):
                feature.measurement = None
                mapped_descriptors[i,:] = feature.descriptor
                point = (int(feature.prediction[0]),int(feature.prediction[1]))

            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(mapped_descriptors,self.new_descriptors,k=1)

            for f_id, feature in enumerate(self.mapped_features):
                f_matches = matches[f_id]
                feature.missed += 1
                if len(f_matches) > 0:
                    match = f_matches[0]
                    ym=np.array([self.new_keypoints[match.trainIdx].pt]).T
                    if norm(ym-feature.prediction) <= 15:
                        self.draw_mapped(match)
                        feature.missed = 0
                        # self.draw_search(feature)
                        feature.measurement = ym



    def register_image(self,input_image,mapped_features,camera_pose):

        self.mapped_features = mapped_features
        self.camera_pose = camera_pose

        # convert image to black and white (if necessary)
        # and save or create the color copy for use as output
        self.define_new_img(input_image)

        # use this image as a keyframe if needed
        new_kf_declared = self.save_as_keyframe()

        candidates = []

        # if features can be reliably matched in this image
        if self.ready_for_matching():

            self.new_keypoints, self.new_descriptors = self.get_orb_n(500,self.new_image)

            # draw keyframe image above new image
            self.create_output_image_base()

            self.find_mapped()
            candidates = self.find_new_matches()


        self.update_inlier_history(len(candidates))
        if self.declare_new_keyframe():
            self.new_kf_desired = True

        self.draw_kf_keypoints()
        return candidates,new_kf_declared

    def draw_search(self,feature):
        vmax = self.output_image.shape[0]/2
        x,y = feature.prediction.flatten()
        r = int(feature.uncertainty_radius)
        cv2.circle(self.output_image, (int(x),vmax+int(y)), r, (0, 0, 255), 1)

    def draw_unmapped(self,matches):
        vmax = self.output_image.shape[0]/2
        for match in matches:
            kf_keypoint = match.queryIdx
            new_keypoint = match.trainIdx
            (x1,y1) = self.kf_keypoints[kf_keypoint].pt
            (x2,y2) = self.new_keypoints[new_keypoint].pt
            cv2.line(self.output_image, (int(x1),int(y1)), (int(x2),vmax+int(y2)), (255, 0, 255), 2)

    def draw_mapped(self,match):
        vmax = self.output_image.shape[0]/2
        new_keypoint = match.trainIdx
        (x2,y2) = self.new_keypoints[new_keypoint].pt
        cv2.circle(self.output_image, (int(x2),vmax+int(y2)), 4, (0, 255, 0), -1)


    def draw_kf_keypoints(self):
        for kp in self.kf_keypoints:
            (x2,y2) = kp.pt
            cv2.circle(self.output_image, (int(x2),int(y2)), 3, (255, 0, 255), 1)
