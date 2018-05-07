import numpy as np
from numpy.linalg import norm

for feature_id,feature in estimator.feature_database.initialized.iteritems():
    print norm(feature.global_position-odb.idb.reference.features_xyz[:,feature_id:feature_id+1])
    