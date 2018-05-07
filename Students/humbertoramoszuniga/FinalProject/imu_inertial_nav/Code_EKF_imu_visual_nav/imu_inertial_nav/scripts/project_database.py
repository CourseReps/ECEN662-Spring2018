from rpmsat import Database


class Reference(object):
    def __init__(self):
        self.kinematics = None
        self.pose = None


class InputDatabase(Database):
    def __init__(self, idb_path):
        self.reference = Reference()

        self.target_attitude = None

        self.imu = None
        self.camera = None
        self.feature_detector = None

        super(InputDatabase, self).__init__(idb_path)
