import numpy as np

# convert a vector to equivalent skew symmetric representation
def skew_symmetric(V):
    vector = np.array(V).flatten()
    v1,v2,v3=vector
    skew_symmetric=np.array([ [  0,-v3, v2],
                              [ v3,  0,-v1],
                              [-v2, v1,  0] ])
    return skew_symmetric

def col_vector(v):
    """
    Returns input vector as column vetor.

    If v is a 1D array, 2D column vector, or 2D row vector, this function
    will return a 2D column vector. Otherwise, it will return the original input.
    """
    shape_v = np.shape(v)

    # v is 1D array
    if len(shape_v) == 1:
        return np.array([v]).T

    # v is a 2D row vector
    elif shape_v[0] == 1:
        return v.T

    # v is a 2D col vector
    elif shape_v[1] == 1:
        return v

    # v is not a vector
    else:
        return v