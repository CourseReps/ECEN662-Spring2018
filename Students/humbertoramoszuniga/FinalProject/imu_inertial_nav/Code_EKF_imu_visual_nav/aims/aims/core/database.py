import os
import numpy as np
from abc import ABCMeta,abstractmethod

def database_from_file(filename):
    import cPickle as pickle
    open_file = open( os.path.join(os.environ['HOME'],filename), "rb" )
    return pickle.load( open_file)

class Database(object):

    def __init__(self,databse_path):
        self.path = os.path.join(os.environ['HOME'],databse_path)

    def write_to_file(self):
        import cPickle as pickle
        pickle.dump(self,open( self.path, "wb" ))



class Reference(object):

    def __init__(self):
        self.kinematics = None
        self.features_xyz = None
        self.system_params = None
        self.pose = None

class FeatureCount(object):

        def __init__(self,length):
            self.features_in_map = np.zeros(length,dtype=int)
            self.visible_features = np.zeros(length,dtype=int)

class InputDatabase(Database):

    def __init__(self,idb_path):
        self.reference = Reference()

        self.imu = None
        self.rangefinder = None
        self.camera = None
        self.gps = None
        self.feature_detector = None
        self.system_params = None

        super(InputDatabase,self).__init__(idb_path)


class OutputDatabase(Database):

    def __init__(self,odb_path):
        self.idb = None
        self.estimate_history = None
        self.three_sigma = None
        self.cloned_pose_history = None
        self.feature_count = None
        super(OutputDatabase,self).__init__(odb_path)
