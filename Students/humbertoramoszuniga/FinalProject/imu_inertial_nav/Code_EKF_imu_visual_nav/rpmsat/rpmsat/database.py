import os
import numpy as np

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
